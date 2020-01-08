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
class Profiler{
    chrono::steady_clock::time_point startT;
    clock_t begin_time;
    public:
    void start(){
        startT=chrono::steady_clock::now();
        begin_time = clock();
    }
    void check(int i){
        float cpu_millis = float( clock() -begin_time ) / CLOCKS_PER_SEC* 1000;
        float RT = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-startT).count();
        float ratio=cpu_millis/float(RT);
        ratio=(8-ratio)*RT;
        std::cout<<"CHECK "<<i<<": "<<ratio<<std::endl;
        startT=chrono::steady_clock::now();
        begin_time=clock();
    }
    void elapsed(int i){

        float RT = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now()-startT).count();
        std::cout<<"ELAPSED "<<i<<": "<<RT<<std::endl;
        startT=chrono::steady_clock::now();
        begin_time=clock();
    }
};
class Pointnet{
    private:
        Session* session;
        Status status;
        ros::Time r_time;
        float scaleI;
        bool gfeat;
        Profiler prof;
    public:
        float RANGE=130.0;
        int  num=15000;
        bool VERBOSE=false;
        ros::Publisher pub;
        bool dumb_label;
        Pointnet(ros::NodeHandle& nh, std::string topic_out, float sc, bool gfeat, bool dumb_label):scaleI(sc), gfeat(gfeat),dumb_label(dumb_label)
        {
            pub=nh.advertise<pcl::PointCloud<pcl::PointXYZI> >(topic_out, 1);
            // Initialize a tensorflow session
            auto options = tensorflow::SessionOptions();
            options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.2);
            options.config.mutable_gpu_options()->set_allow_growth(true);

            status = NewSession(options, &session);
            if (!status.ok()) {
              std::cout << status.ToString() << "\n";
            }
            // Read in the protobuf graph we exported
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
        }
        ~Pointnet(){
            session->Close();
        }

        void callback(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_msg)
        {
            if(VERBOSE)
                prof.start();
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_mv(new pcl::PointCloud<pcl::PointXYZI>);
            //pcl::copyPointCloud(*cloud_msg, *cloud_mv);
            int npoints = cloud_msg->points.size();


            Tensor nPointsSample(DT_INT32, TensorShape({1}));
            nPointsSample.vec<int>()(0)=num;
            Tensor MaxRange(DT_FLOAT, TensorShape({1}));
            MaxRange.vec<float>()(0)=RANGE;
            Tensor cloud_inT(DT_FLOAT, {npoints, 4});
            auto itm = cloud_inT.tensor<float,2>();
            for(int i=0;i<npoints; i++)
            {
                itm(i, 0)=cloud_msg->points[i].x;
                itm(i, 1)=cloud_msg->points[i].y;
                itm(i, 2)=cloud_msg->points[i].z;
                itm(i, 3)=cloud_msg->points[i].intensity/scaleI;
            }
            std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
                {"MaxRange", MaxRange},
                { "NumSample", nPointsSample },
                { "cloud_in", cloud_inT }
            };


            // The session will initialize the outputs
            std::vector<tensorflow::Tensor> outputs;

            // Run the session, evaluating our "c" operation from the graph
            //status = session->Run(inputs, {"cloud_out", "layer1/points_out"}, {}, &outputs);
            if(gfeat)
                status = session->Run(inputs, {"cloud_out", "gfeat"}, {}, &outputs);
            else
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
            //
            if(gfeat)//Encode into first 256 points :-)
            {
                auto dat = outputs[1].tensor<float,2>();
                for(int i=0;i<256;i++)
                {
                    pcl::PointXYZI p;
                    p.x=dat(1,i*4);
                    p.y=dat(1,i*4+1);
                    p.z=dat(1,i*4+2);
                    p.intensity=dat(1,i*4+3);
                    cloud_mv->points.push_back(p);
                }
            }
            int nPoints=cloud_outT.dimension(0);
            if(VERBOSE)
                std::cout<<"NPOINTSOUT: "<<nPoints<<std::endl;
            for(int i=0;i<nPoints;i++)
            {
                pcl::PointXYZI p;
                p.x=cloud_outT(i,0);
                p.y=cloud_outT(i,1);
                p.z=cloud_outT(i,2);
                if(dumb_label)
                    p.intensity=0;
                else
                    p.intensity=cloud_outT(i,3);
                cloud_mv->points.push_back(p);
            }
            //std::cout << output_c() << "\n"; // 30
            //std::cout<<cloud_mv->points.size()<<std::endl;
            //clock_t begin_time = clock();
            cloud_mv->header.stamp= cloud_msg->header.stamp;
            //pcl_conversions::toPCL(r_time,cloud_mv->header.stamp);
            cloud_mv->header.frame_id="velodyne";
            pub.publish(cloud_mv);
            if(VERBOSE)
                prof.elapsed(0);
        }
};

int main(int argc, char** argv)
{
	string p_dir;
    string topic_in="velodyne_points";
    string topic_out="semantic";
    float scaleI=1;
    float range=130;
    int num=15000;
    bool dumb_label=false;

	po::options_description desc("Allowed options");
	desc.add_options()
	("help,h", "produce help message")
    ("scale,s", po::value<float>(&scaleI),"Pointnet expects Intensity in [0,1]. Devide by s to scale.")
    ("range,r", po::value<float>(&range),"Max range.")
    ("num,n", po::value<int>(&num),"Num Points.")
    ("gfeat,g", "Publish global feature vector.")
    ("verbose,v", "Show msec.")
    ("dumb,d", "Publish points with dumb label (All 0).")
    ("topic_out,t", po::value<string>(&topic_out),"Topic to publish")
	("topic_in,p", po::value<string>(&topic_in), "Point cloud files");

	po::parsed_options parsed_options = po::command_line_parser(argc, argv)
      .options(desc).style(po::command_line_style::unix_style ).run();
    po::variables_map vm;
	po::store(parsed_options, vm);
    po::notify(vm);
	if(vm.count("help"))
        { cout<<desc; return 0; }
    bool gfeat=false;
	if(vm.count("gfeat"))
        gfeat=true;
	if(vm.count("dumb"))
        dumb_label=true;

	ros::init (argc,argv,"pointnet_node");
	ros::NodeHandle nh;
    Pointnet pointnet(nh, topic_out, scaleI, gfeat,dumb_label);
	if(vm.count("verbose"))
        pointnet.VERBOSE=true;
    pointnet.RANGE=range;
    pointnet.num=num;
    ros::Subscriber sub = nh.subscribe(topic_in, 1, &Pointnet::callback, &pointnet);
    ros::spin();
}
