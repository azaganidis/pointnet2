#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/io.h>

#include <iostream>
#include <ctime>
#include <boost/program_options.hpp>
#include <fstream>

using namespace tensorflow;
using namespace std;
namespace po = boost::program_options;
class Pointnet{
    private:
        Session* session;
        Status status;
        ros::Time r_time;
        std::fstream feature_out;
    public:
        ros::Publisher pub;
        Pointnet(ros::NodeHandle& nh, std::string topic_out){
            pub=nh.advertise<pcl::PointCloud<pcl::PointXYZI> >(topic_out, 1);
            // Initialize a tensorflow session
            status = NewSession(SessionOptions(), &session);
            if (!status.ok()) {
              std::cout << status.ToString() << "\n";
            }
            // Read in the protobuf graph we exported
            // (The path seems to be relative to the cwd. Keep this in mind
            // when using `bazel run` since the cwd isn't where you call
            // `bazel run` but from inside a temp folder.)
            GraphDef graph_def;
            status = ReadBinaryProto(Env::Default(), "models/pointnet.pb", &graph_def);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
            }
            // Add the graph to the session
            status = session->Create(graph_def);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
            }
            feature_out.open ("feature_out.txt", std::fstream::in | std::fstream::out | std::fstream::app);
        }
        ~Pointnet(){
            session->Close();
        }

        void callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_msg)
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_mv(new pcl::PointCloud<pcl::PointXYZI>);
            //pcl::copyPointCloud(*cloud_msg, *cloud_mv);
            int npoints = cloud_msg->points.size();


            Tensor cloud_inT(DT_FLOAT, {npoints, 4});
            auto itm = cloud_inT.tensor<float,2>();
            for(int i=0;i<npoints; i++)
            {
                itm(i, 0)=cloud_msg->points[i].x;
                itm(i, 1)=cloud_msg->points[i].y;
                itm(i, 2)=cloud_msg->points[i].z;
                itm(i, 3)=cloud_msg->points[i].intensity;
            }
            std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                { "cloud_in", cloud_inT }
            };


            // The session will initialize the outputs
            std::vector<tensorflow::Tensor> outputs;

            // Run the session, evaluating our "c" operation from the graph
            //status = session->Run(inputs, {"cloud_out", "layer1/points_out"}, {}, &outputs);
            status = session->Run(inputs, {"cloud_out"}, {}, &outputs);
            if (!status.ok()) {
                std::cout << status.ToString() << "\n";
            }
            // Grab the first output (we only evaluated one graph node: "c")
            // and convert the node to a scalar representation.
            //auto output_c = outputs[0].tensor<float>();

            // (There are similar methods for vectors and matrices here:
            // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

            // Print the results
            //std::cout << outputs[1].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
            auto cloud_outT = outputs[0].tensor<float, 2> ();
            //cloud_mv->points.clear();
            for(int i=0;i<15000;i++)
            {
                pcl::PointXYZI p;
                p.x=cloud_outT(i,0);
                p.y=cloud_outT(i,1);
                p.z=cloud_outT(i,2);
                p.intensity=cloud_outT(i,3);
                cloud_mv->points.push_back(p);
            }
            /*
            auto points_outT = outputs[1].tensor<float, 3> ();
            for(int i=0;i<1024;i++)
            {
                feature_out<<points_outT(0,i,2)<<" "<<points_outT(0,i,3)<<std::endl;
            }
            */
            //std::cout << output_c() << "\n"; // 30
            //std::cout<<cloud_mv->points.size()<<std::endl;

            //clock_t begin_time = clock();
            r_time = ros::Time::now();
            pcl_conversions::toPCL(r_time,cloud_mv->header.stamp);
            cloud_mv->header.frame_id="velodyne";
            pub.publish(cloud_mv);
        }
};

int main(int argc, char** argv)
{
	string p_dir;
    string topic_in="velodyne_points";
    string topic_out="semantic";

	po::options_description desc("Allowed options");
	desc.add_options()
	("help,h", "produce help message")
    ("topic_out,t", po::value<string>(&topic_out),"Topic to publish")
	("topic_in,p", po::value<string>(&topic_in), "Point cloud files");

	po::parsed_options parsed_options = po::command_line_parser(argc, argv)
      .options(desc).style(po::command_line_style::unix_style ).run();
    po::variables_map vm;
	po::store(parsed_options, vm);
    po::notify(vm);
	if(vm.count("help"))
        { cout<<desc; return 0; }

	ros::init (argc,argv,"pointnet_node");
	ros::NodeHandle nh;
    Pointnet pointnet(nh, topic_out);
    ros::Subscriber sub = nh.subscribe(topic_in, 1, &Pointnet::callback, &pointnet);
    ros::spin();
}
