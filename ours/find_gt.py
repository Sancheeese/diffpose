import numpy as np
import torch
import matplotlib

# matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
from diffpose.calibration import RigidTransform
# from ours.utils.CT_dataset import IntubationDataset, Transforms
from ours.case.cyx.CT_dataset import IntubationDataset, Transforms
# from ours.case.yjj.CT_dataset import IntubationDataset, Transforms
# from ours.case.yjj.CT_dataset2 import IntubationDataset, Transforms
from ours.utils.drr import DRR
import os


class ConsolePoseAdjuster:
    def __init__(self, drr, specimen, transforms, device="cuda:0", save_dir="./pose_adjust_images"):
        self.drr = drr
        self.transforms = transforms
        self.device = device
        self.specimen = specimen
        # self.pose_params = [1.8634, -0.7295, -0.5676, 265.5873, 15.4750, 107.0555]
        # pose_params = [1.8524, -0.8056, -0.5666, 256.6452, 3.0481, 128.6834]
        # cyx
        pose_params = [1.7459615468978882, 0.756145179271698, 0.22575482726097107, 278.4220275878906, 236.57859802246094, 81.5419921875]
        # self.pose_params = [0, 0, 0, 0, 0, 0]
        # cyx
        self.pose_params = [0.1, -0.1, -0.1, -15, 0, 15]
        # jjl
        # self.pose_params = [-0.55, 0.15, 0.15, -30, 270, 0]
        # xyl
        # self.pose_params = [-0.6, 0.125, 0.25, 50, 385, 10]
        # self.pose_params = [-0.3, 0.2, -0.05, 60, 280, 60]

        rot = torch.tensor([pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32, device=self.device)
        self.pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        self.rotation_step = 0.025
        self.translation_step = 5
        self.save_dir = save_dir
        self.step_count = 0

        self.center_pose = specimen.center_pose.to(device)
        self.back_pose = specimen.back_pose.to(device)
        self.isocenter_pose = specimen.isocenter_pose.to(device)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

    def update_pose(self, param_idx, delta):
        self.pose_params[param_idx] += delta
        self.step_count += 1
        self.print_current_pose()
        self.save_current_image()

    def print_current_pose(self):
        print("\n" + "=" * 60)
        print(f"步骤 {self.step_count} - 当前位姿参数:")
        print(f"旋转 - alpha(X): {self.pose_params[0]:.4f} rad")
        print(f"旋转 - beta(Y): {self.pose_params[1]:.4f} rad")
        print(f"旋转 - gamma(Z): {self.pose_params[2]:.4f} rad")
        print(f"平移 - bx(X): {self.pose_params[3]:.2f} mm")
        print(f"平移 - by(Y): {self.pose_params[4]:.2f} mm")
        print(f"平移 - bz(Z): {self.pose_params[5]:.2f} mm")
        print("=" * 60)

    def get_current_pose(self):
        rot = torch.tensor([self.pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([self.pose_params[3:]], dtype=torch.float32, device=self.device)
        pose = RigidTransform(rot, xyz, "euler_angles", "ZYX").to(self.device)
        # pose = self.isocenter_pose.compose(self.center_pose).compose(pose).compose(self.back_pose)
        # pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        # return self.isocenter_pose
        # return self.pose
        return pose.compose(self.pose)

    def generate_drr_image(self):
        current_pose = self.get_current_pose()
        with torch.no_grad():
            pred_img = self.drr(None, None, None, pose=current_pose)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
            img_np = pred_img.squeeze().cpu().numpy()
            return img_np

    def save_current_image(self):
        """保存当前位姿对应的DRR图像"""
        img_np = self.generate_drr_image()

        plt.figure(figsize=(12, 8))
        plt.imshow(img_np, cmap='gray')
        plt.title(f'DRR Image - Step {self.step_count}')
        plt.axis('off')

        # 添加参数信息
        param_text = (f"alpha: {self.pose_params[0]:.3f} rad\n"
                      f"beta: {self.pose_params[1]:.3f} rad\n"
                      f"gamma: {self.pose_params[2]:.3f} rad\n"
                      f"bx: {self.pose_params[3]:.1f} mm\n"
                      f"by: {self.pose_params[4]:.1f} mm\n"
                      f"bz: {self.pose_params[5]:.1f} mm")

        plt.figtext(0.02, 0.98, param_text, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # 添加控制说明
        control_text = ("控制命令:\n"
                        "Q/W: alpha +/-\n"
                        "E/A: beta +/-\n"
                        "S/D: gamma +/-\n"
                        "U/J: bx +/-\n"
                        "I/K: by +/-\n"
                        "O/L: bz +/-\n"
                        "0: 退出")

        plt.figtext(0.75, 0.98, control_text, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        filename = f"{self.save_dir}/pose_step_{self.step_count:03d}.png"
        # plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ 图像已保存: {filename}")

    def show_help(self):
        print("\n🎮 控制命令:")
        print("┌─────────────────┬─────────────────┐")
        print("│    命令         │       功能      │")
        print("├─────────────────┼─────────────────┤")
        print("│      Q          │   alpha +0.05   │")
        print("│      W          │   alpha -0.05   │")
        print("│      E          │   beta +0.05    │")
        print("│      A          │   beta -0.05    │")
        print("│      S          │   gamma +0.05   │")
        print("│      D          │   gamma -0.05   │")
        print("│      U          │   bx +5.0       │")
        print("│      J          │   bx -5.0       │")
        print("│      I          │   by +5.0       │")
        print("│      K          │   by -5.0       │")
        print("│      O          │   bz +5.0       │")
        print("│      L          │   bz -5.0       │")
        print("│      H          │   显示帮助      │")
        print("│      0          │   退出程序      │")
        print("└─────────────────┴─────────────────┘")

    def run(self):
        print("🚀 开始交互式位姿调整")
        print(f"📁 图像将保存到: {os.path.abspath(self.save_dir)}")
        self.show_help()

        # 保存初始图像
        self.save_current_image()

        while True:
            try:
                print("\n" + "─" * 50)
                cmd = input("🎯 请输入命令 (H显示帮助): ").strip().lower()

                if not cmd:
                    continue

                if cmd == '0':  # 退出键改为0
                    print("👋 退出程序")
                    break
                elif cmd == 'h':
                    self.show_help()
                elif cmd == 'q':  # Q: alpha +
                    self.update_pose(0, self.rotation_step)
                elif cmd == 'a':  # W: alpha -
                    self.update_pose(0, -self.rotation_step)
                elif cmd == 'w':  # E: beta +
                    self.update_pose(1, self.rotation_step)
                elif cmd == 's':  # A: beta -
                    self.update_pose(1, -self.rotation_step)
                elif cmd == 'e':  # S: gamma +
                    self.update_pose(2, self.rotation_step)
                elif cmd == 'd':  # D: gamma -
                    self.update_pose(2, -self.rotation_step)
                elif cmd == 'u':  # U: bx +
                    self.update_pose(3, self.translation_step)
                elif cmd == 'j':  # J: bx -
                    self.update_pose(3, -self.translation_step)
                elif cmd == 'i':  # I: by +
                    self.update_pose(4, self.translation_step)
                elif cmd == 'k':  # K: by -
                    self.update_pose(4, -self.translation_step)
                elif cmd == 'o':  # O: bz +
                    self.update_pose(5, self.translation_step)
                elif cmd == 'l':  # L: bz -
                    self.update_pose(5, -self.translation_step)
                else:
                    print("❌ 无效命令，输入 'h' 查看帮助")

            except KeyboardInterrupt:
                print("\n🛑 程序被用户中断")
                break
            except Exception as e:
                print(f"💥 发生错误: {e}")

        print(f"✅ 程序结束，共生成 {self.step_count} 张图像")
        print(f"📂 图像保存在: {os.path.abspath(self.save_dir)}")


def main():
    device = "cuda:1"

    # 初始化DRR
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/CT/ZhangYanLing/20240318122424.893/203"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/张燕玲/ERCP/YANLING^ZHANG^/20240311150042/1"
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/CT/ChenYuXin/20240530105749/304"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/ERCP/YUXING^CHEN^/20240531155445/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨嘉洁/CT/YangJiaJie/20240301222402.121000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/杨嘉洁/ERCP/JIAJIE^YANG^/20240304161225/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/CT/ShiJianJi/20231019091100.063/602"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/史建基/ERCP/SHI^JIANJI^/20231023150900/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉露/CT/XuYuLu/20240326172117.697000/3"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/许玉露/ERCP/YULU^XU^/20240403154139/1"
    # root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/江莲娇/CT/JiangLianJiao/20240716162518.679/602"
    # x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/江莲娇/ERCP/JIANGLIANJIAO^^/20240724145013/1"
    # specimen = IntubationDataset(root, x_root, z_offset=100, factors=[0.5, 4, 0.5])
    # specimen = IntubationDataset(root, x_root, y_offset=155, z_cut=600, factors=[3, 0.5, 0.5])
    specimen = IntubationDataset(root, x_root, z_offset=70)
    # specimen = IntubationDataset(root, x_root, y_offset=-100, z_offset=100, factors=[0.5, 4, 0.5])
    # specimen = IntubationDataset(root, x_root, z_offset=50, factors=[5, 1.5, 2])
    # specimen = IntubationDataset(root, x_root, y_offset=-150, z_offset=400, factors=[0.5, 4, 0.5])
    # specimen = IntubationDataset(root, x_root, x_offset=-60, z_offset=150, factors=[0.5, 8, 0.5])

    height = 256
    subsample = 512 / height
    delx = specimen.delx * subsample

    drr = DRR(
        specimen.volume,
        specimen.spacing,
        sdr=specimen.sdr,
        height=height,
        delx=delx,
        reverse_x_axis=True,
        bone_attenuation_multiplier=6,
    ).to(device)

    transforms = Transforms(height)

    img, pose = specimen[33]
    img = transforms(img, reverse=False).to(device).to(torch.float32)
    plt.figure()
    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.show()

    # 创建调整器
    adjuster = ConsolePoseAdjuster(drr, specimen, transforms, device, save_dir="./pose_adjust_results")

    # 运行交互式调整
    adjuster.run()


if __name__ == "__main__":
    main()