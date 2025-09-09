import os
import h5py
import imageio


def create_video_for_image_obs(demos, image_obs_name, video_folder):
    video_fn = os.path.join(video_folder, f"{image_obs_name}.mp4")
    writer = imageio.get_writer(video_fn, fps=20)

    #Iterate over all demos
    for demo in demos:
        demo_images = demos[demo][f"obs/{image_obs_name}"][:]
        # Write the video for this camera
        for i in range(demo_images.shape[0]):
            writer.append_data(demo_images[i])

    writer.close()


def hdf_to_videos(demo_fn, video_folder, task_name):
    ### With New Setup

    output_folder_name = task_name
    video_folder = os.path.join(video_folder,output_folder_name)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    print(f"Saving videos to : {video_folder}")
    print("==================================================================================================")

    demo_file = h5py.File(demo_fn)
    demos = demo_file['data']

    # First, find all image obs names
    image_obs_list = []
    for demo in demos:
        demo = demos[demo]
        obs = demo['obs']
        for mod in obs:
            # print(mod)
            if "image" in mod or "rgb" in mod:
                image_obs_list.append(mod)
        break

    # Create video for each camera
    for image_obs in image_obs_list:
        print(f"Processing camera: {image_obs}")
        create_video_for_image_obs(demos, image_obs, video_folder)


if __name__ == "__main__":

    demo = "/home/mbronars/zhenyang/demos/stacking_cup_demo_0112/demo.hdf5" # "/media/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"
    # save_dir = "/media/nadun/Data/phd_project/robomimic/videos/full_dataset_videos"
    demo = "/coc/flash7/zhenyang/data/robomimic-sim/square_skill_image.hdf5"
    
    save_dir = "/coc/flash7/zhenyang/data/robomimic-sim/square_videos"

    task_name = "square_low_dim_skill_dataset"

    hdf_to_videos(demo, save_dir, task_name)