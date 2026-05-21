#include <cv_bridge/cv_bridge.h>
#include <boost/bind/bind.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

namespace
{
struct RawPairSample
{
  std::uint64_t sample_id{};
  sensor_msgs::ImageConstPtr msg_a;
  sensor_msgs::ImageConstPtr msg_b;
  double stamp_a{};
  double stamp_b{};
};

struct DecodedPairSample
{
  std::uint64_t sample_id{};
  cv::Mat image_a;
  cv::Mat image_b;
  double stamp_a{};
  double stamp_b{};
};

struct SaveJob
{
  int index{};
  cv::Mat image_a;
  cv::Mat image_b;
  double stamp_a{};
  double stamp_b{};
  std::string filename_a;
  std::string filename_b;
};

struct PairMetadata
{
  int pair_index{};
  std::string camera_a_image;
  std::string camera_b_image;
  double camera_a_stamp{};
  double camera_b_stamp{};
  double timestamp_delta_seconds{};
};

std::string getStringParam(const ros::NodeHandle& nh, const std::string& key, const std::string& fallback)
{
  std::string value;
  if (nh.getParam(key, value))
  {
    return value;
  }
  return fallback;
}

cv::Mat convertToBgr(const sensor_msgs::ImageConstPtr& msg)
{
  const std::string& encoding = msg->encoding;
  if (encoding == "bgr8")
  {
    return cv_bridge::toCvShare(msg, encoding)->image;
  }
  if (encoding == "rgb8")
  {
    cv::Mat bgr;
    cv::cvtColor(cv_bridge::toCvShare(msg, encoding)->image, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  if (encoding == "mono8")
  {
    cv::Mat bgr;
    cv::cvtColor(cv_bridge::toCvShare(msg, encoding)->image, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }
  return cv_bridge::toCvCopy(msg, "bgr8")->image;
}
}  // namespace

class PairCaptureNode
{
public:
  PairCaptureNode()
    : nh_(),
      pnh_("~"),
      it_(pnh_)
  {
    topic_a_ = getStringParam(pnh_, "camera_a/image_topic", "/camera_a/color/image_raw");
    topic_b_ = getStringParam(pnh_, "camera_b/image_topic", "/camera_b/color/image_raw");
    pnh_.param("synchronization/queue_size", queue_size_, 5);
    pnh_.param("synchronization/slop_seconds", slop_seconds_, 0.03);
    pnh_.param("synchronization/subscriber_queue_size", subscriber_queue_size_, 1);
    pnh_.param("capture/preview_fps", preview_fps_, 10.0);
    pnh_.param("capture/preview_width", preview_width_, 960);
    pnh_.param("capture/preview_only", preview_only_, false);
    pnh_.param("capture/image_extension", image_extension_, std::string(".png"));
    pnh_.param("capture/png_compression", png_compression_, 3);
    pnh_.param("capture/jpeg_quality", jpeg_quality_, 95);
    pnh_.param("capture/session_name", session_name_, std::string("sample_session"));
    pnh_.param("capture/output_root", output_root_, std::string("/tmp/two_camera_rgb_extrinsic"));
    pnh_.param("preview/publish_topics", publish_preview_topics_, true);

    dataset_dir_ = output_root_ + "/" + session_name_;
    camera_a_dir_ = dataset_dir_ + "/camera_a";
    camera_b_dir_ = dataset_dir_ + "/camera_b";
    metadata_path_ = dataset_dir_ + "/pairs_metadata.json";

    ensureDirectory(camera_a_dir_);
    ensureDirectory(camera_b_dir_);

    sub_a_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh_, topic_a_, subscriber_queue_size_);
    sub_b_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh_, topic_b_, subscriber_queue_size_);
    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(queue_size_), *sub_a_, *sub_b_);
    sync_->setMaxIntervalDuration(ros::Duration(slop_seconds_));
    sync_->registerCallback(boost::bind(&PairCaptureNode::syncCallback, this, boost::placeholders::_1, boost::placeholders::_2));

    if (publish_preview_topics_)
    {
      preview_pub_a_ = it_.advertise("preview/camera_a", 1);
      preview_pub_b_ = it_.advertise("preview/camera_b", 1);
    }

    save_thread_ = std::thread(&PairCaptureNode::saveWorkerLoop, this);
  }

  ~PairCaptureNode()
  {
    {
      std::lock_guard<std::mutex> lock(save_mutex_);
      saver_stopping_ = true;
    }
    save_cv_.notify_all();
    if (save_thread_.joinable())
    {
      save_thread_.join();
    }
  }

  void run()
  {
    if (!preview_only_)
    {
      cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    }

    ros::Rate rate(std::max(1.0, preview_fps_));
    ROS_INFO("Capture node started. preview_only=%s", preview_only_ ? "true" : "false");

    while (ros::ok() && !shutdown_requested_)
    {
      updatePreviewIfNeeded();
      if (!preview_only_)
      {
        handleUi();
      }
      rate.sleep();
    }

    writeMetadata();
    if (!preview_only_)
    {
      cv::destroyAllWindows();
    }
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  image_transport::Publisher preview_pub_a_;
  image_transport::Publisher preview_pub_b_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_a_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_b_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::string topic_a_;
  std::string topic_b_;
  int queue_size_{5};
  double slop_seconds_{0.03};
  int subscriber_queue_size_{1};
  double preview_fps_{10.0};
  int preview_width_{960};
  bool preview_only_{false};
  bool publish_preview_topics_{true};
  std::string image_extension_{".png"};
  int png_compression_{3};
  int jpeg_quality_{95};
  std::string session_name_;
  std::string output_root_;
  std::string dataset_dir_;
  std::string camera_a_dir_;
  std::string camera_b_dir_;
  std::string metadata_path_;
  std::string window_name_{"Two-Camera RGB Capture"};

  std::mutex sample_mutex_;
  std::optional<RawPairSample> latest_raw_sample_;
  std::optional<DecodedPairSample> latest_decoded_sample_;
  cv::Mat latest_preview_;
  std::uint64_t received_count_{0};
  std::uint64_t saved_count_{0};
  std::uint64_t last_previewed_sample_id_{0};

  std::vector<PairMetadata> metadata_;
  std::queue<SaveJob> save_queue_;
  std::mutex save_mutex_;
  std::condition_variable save_cv_;
  std::thread save_thread_;
  bool saver_stopping_{false};
  bool shutdown_requested_{false};

  void syncCallback(const sensor_msgs::ImageConstPtr& msg_a, const sensor_msgs::ImageConstPtr& msg_b)
  {
    std::lock_guard<std::mutex> lock(sample_mutex_);
    ++received_count_;
    latest_raw_sample_ = RawPairSample{received_count_, msg_a, msg_b, msg_a->header.stamp.toSec(), msg_b->header.stamp.toSec()};
  }

  void updatePreviewIfNeeded()
  {
    std::optional<RawPairSample> raw;
    {
      std::lock_guard<std::mutex> lock(sample_mutex_);
      raw = latest_raw_sample_;
    }
    if (!raw.has_value() || raw->sample_id == last_previewed_sample_id_)
    {
      return;
    }

    cv::Mat image_a = convertToBgr(raw->msg_a);
    cv::Mat image_b = convertToBgr(raw->msg_b);

    latest_decoded_sample_ = DecodedPairSample{raw->sample_id, image_a, image_b, raw->stamp_a, raw->stamp_b};
    publishPreviewImages(*latest_decoded_sample_);

    if (!preview_only_)
    {
      latest_preview_ = buildPreview(*latest_decoded_sample_);
    }
    last_previewed_sample_id_ = raw->sample_id;
  }

  void publishPreviewImages(const DecodedPairSample& sample)
  {
    if (!publish_preview_topics_)
    {
      return;
    }
    cv::Mat preview_a = resizeToWidth(sample.image_a, std::max(320, preview_width_ / 2));
    cv::Mat preview_b = resizeToWidth(sample.image_b, std::max(320, preview_width_ / 2));

    std_msgs::Header header_a;
    header_a.stamp = ros::Time(sample.stamp_a);
    header_a.frame_id = "camera_a_preview";
    std_msgs::Header header_b;
    header_b.stamp = ros::Time(sample.stamp_b);
    header_b.frame_id = "camera_b_preview";

    preview_pub_a_.publish(cv_bridge::CvImage(header_a, "bgr8", preview_a).toImageMsg());
    preview_pub_b_.publish(cv_bridge::CvImage(header_b, "bgr8", preview_b).toImageMsg());
  }

  cv::Mat buildPreview(const DecodedPairSample& sample) const
  {
    cv::Mat panel_a = annotate(sample.image_a, "Camera A RGB");
    cv::Mat panel_b = annotate(sample.image_b, "Camera B RGB");
    const int width_each = std::max(1, preview_width_ / 2);
    panel_a = resizeToWidth(panel_a, width_each);
    panel_b = resizeToWidth(panel_b, width_each);
    const int target_height = std::min(panel_a.rows, panel_b.rows);
    panel_a = resizeToHeight(panel_a, target_height);
    panel_b = resizeToHeight(panel_b, target_height);
    cv::Mat preview;
    cv::hconcat(panel_a, panel_b, preview);

    std::ostringstream status;
    status << "Received pairs: " << received_count_ << " | Saved pairs: " << saved_count_ << " | Keys: S save, Q quit";
    cv::putText(preview, status.str(), cv::Point(20, std::max(40, preview.rows - 20)),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    return preview;
  }

  void handleUi()
  {
    if (!latest_preview_.empty())
    {
      cv::imshow(window_name_, latest_preview_);
    }
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 'Q')
    {
      shutdown_requested_ = true;
      return;
    }
    if (key == 's' || key == 'S')
    {
      queueCurrentSampleForSave();
    }
  }

  void queueCurrentSampleForSave()
  {
    if (!latest_decoded_sample_.has_value())
    {
      ROS_WARN("No synchronized sample is available to save yet.");
      return;
    }

    const int next_index = static_cast<int>(saved_count_ + 1);
    std::ostringstream name_a;
    std::ostringstream name_b;
    name_a << "camera_a_" << std::setw(4) << std::setfill('0') << next_index << image_extension_;
    name_b << "camera_b_" << std::setw(4) << std::setfill('0') << next_index << image_extension_;

    SaveJob job;
    job.index = next_index;
    job.image_a = latest_decoded_sample_->image_a.clone();
    job.image_b = latest_decoded_sample_->image_b.clone();
    job.stamp_a = latest_decoded_sample_->stamp_a;
    job.stamp_b = latest_decoded_sample_->stamp_b;
    job.filename_a = name_a.str();
    job.filename_b = name_b.str();

    {
      std::lock_guard<std::mutex> lock(save_mutex_);
      save_queue_.push(std::move(job));
    }
    save_cv_.notify_one();

    saved_count_ = next_index;
    metadata_.push_back(
      PairMetadata{
        next_index,
        name_a.str(),
        name_b.str(),
        latest_decoded_sample_->stamp_a,
        latest_decoded_sample_->stamp_b,
        std::abs(latest_decoded_sample_->stamp_a - latest_decoded_sample_->stamp_b)
      });
    ROS_INFO("Queued pair %04d for saving.", next_index);
  }

  void saveWorkerLoop()
  {
    while (true)
    {
      SaveJob job;
      {
        std::unique_lock<std::mutex> lock(save_mutex_);
        save_cv_.wait(lock, [this]() { return saver_stopping_ || !save_queue_.empty(); });
        if (saver_stopping_ && save_queue_.empty())
        {
          return;
        }
        job = std::move(save_queue_.front());
        save_queue_.pop();
      }
      writeImage(camera_a_dir_ + "/" + job.filename_a, job.image_a);
      writeImage(camera_b_dir_ + "/" + job.filename_b, job.image_b);
      ROS_INFO("Saved pair %04d to disk.", job.index);
    }
  }

  void writeImage(const std::string& path, const cv::Mat& image) const
  {
    std::vector<int> params;
    if (endsWith(path, ".png"))
    {
      params = {cv::IMWRITE_PNG_COMPRESSION, png_compression_};
    }
    else if (endsWith(path, ".jpg") || endsWith(path, ".jpeg"))
    {
      params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
    }
    cv::imwrite(path, image, params);
  }

  void writeMetadata() const
  {
    std::ofstream out(metadata_path_);
    out << "{\n";
    out << "  \"session_name\": \"" << session_name_ << "\",\n";
    out << "  \"camera_a_topic\": \"" << topic_a_ << "\",\n";
    out << "  \"camera_b_topic\": \"" << topic_b_ << "\",\n";
    out << "  \"received_pair_count\": " << received_count_ << ",\n";
    out << "  \"saved_pair_count\": " << saved_count_ << ",\n";
    out << "  \"pairs\": [\n";
    for (std::size_t i = 0; i < metadata_.size(); ++i)
    {
      const auto& item = metadata_[i];
      out << "    {\n";
      out << "      \"pair_index\": " << item.pair_index << ",\n";
      out << "      \"camera_a_image\": \"" << item.camera_a_image << "\",\n";
      out << "      \"camera_b_image\": \"" << item.camera_b_image << "\",\n";
      out << "      \"camera_a_stamp\": " << item.camera_a_stamp << ",\n";
      out << "      \"camera_b_stamp\": " << item.camera_b_stamp << ",\n";
      out << "      \"timestamp_delta_seconds\": " << item.timestamp_delta_seconds << "\n";
      out << "    }" << (i + 1 == metadata_.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    ROS_INFO("Wrote metadata to %s", metadata_path_.c_str());
  }

  static cv::Mat resizeToWidth(const cv::Mat& image, int width)
  {
    const double scale = static_cast<double>(width) / static_cast<double>(image.cols);
    const int height = std::max(1, static_cast<int>(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return resized;
  }

  static cv::Mat resizeToHeight(const cv::Mat& image, int height)
  {
    const double scale = static_cast<double>(height) / static_cast<double>(image.rows);
    const int width = std::max(1, static_cast<int>(image.cols * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return resized;
  }

  static cv::Mat annotate(const cv::Mat& image, const std::string& title)
  {
    cv::Mat annotated = image.clone();
    cv::putText(annotated, title, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    return annotated;
  }

  static bool endsWith(const std::string& value, const std::string& suffix)
  {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  static void ensureDirectory(const std::string& path)
  {
    const std::string command = "mkdir -p \"" + path + "\"";
    std::system(command.c_str());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "capture_rgb_pairs_node");
  PairCaptureNode node;
  node.run();
  return 0;
}
