import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from diffpose.calibration import RigidTransform
from ours.case.cyx.CT_dataset import IntubationDataset, Transforms
from ours.utils.drr import DRR
import os


class InteractivePoseAdjuster:
    def __init__(self, drr, transforms, device="cuda:0"):
        self.drr = drr
        self.transforms = transforms
        self.device = device
        # 初始位姿参数
        pose_params = [1.7459615468978882, 0.756145179271698, 0.22575482726097107, 278.4220275878906,
                       236.57859802246094, 81.5419921875]
        self.pose_params = [0, 0, 0, 0, 0, 0]  # 相对调整量
        rot = torch.tensor([pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([pose_params[3:]], dtype=torch.float32, device=self.device)
        self.base_pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        self.rotation_step = 0.05
        self.translation_step = 5.0

        # 设置matplotlib
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # 初始显示
        self.update_display()

    def on_key_press(self, event):
        """键盘事件处理"""
        key = event.key.lower()

        if key == '0':  # 退出
            print("👋 退出程序")
            plt.close()
            return
        elif key == 'q':  # alpha +
            self.update_pose(0, self.rotation_step)
        elif key == 'a':  # alpha -
            self.update_pose(0, -self.rotation_step)
        elif key == 'w':  # beta +
            self.update_pose(1, self.rotation_step)
        elif key == 's':  # beta -
            self.update_pose(1, -self.rotation_step)
        elif key == 'e':  # gamma +
            self.update_pose(2, self.rotation_step)
        elif key == 'd':  # gamma -
            self.update_pose(2, -self.rotation_step)
        elif key == 'u':  # bx +
            self.update_pose(3, self.translation_step)
        elif key == 'j':  # bx -
            self.update_pose(3, -self.translation_step)
        elif key == 'i':  # by +
            self.update_pose(4, self.translation_step)
        elif key == 'k':  # by -
            self.update_pose(4, -self.translation_step)
        elif key == 'o':  # bz +
            self.update_pose(5, self.translation_step)
        elif key == 'l':  # bz -
            self.update_pose(5, -self.translation_step)
        elif key == 'r':  # 重置
            self.pose_params = [0, 0, 0, 0, 0, 0]
            self.update_display()
            print("🔄 位姿已重置")

    def update_pose(self, param_idx, delta):
        """更新位姿参数并刷新显示"""
        self.pose_params[param_idx] += delta
        self.print_current_pose()
        self.update_display()

    def print_current_pose(self):
        """打印当前位姿参数"""
        print("\n" + "=" * 60)
        print("当前位姿参数 (相对调整量):")
        print(f"旋转 - alpha(X): {self.pose_params[0]:.4f} rad")
        print(f"旋转 - beta(Y): {self.pose_params[1]:.4f} rad")
        print(f"旋转 - gamma(Z): {self.pose_params[2]:.4f} rad")
        print(f"平移 - bx(X): {self.pose_params[3]:.2f} mm")
        print(f"平移 - by(Y): {self.pose_params[4]:.2f} mm")
        print(f"平移 - bz(Z): {self.pose_params[5]:.2f} mm")

        # 计算绝对位姿
        current_pose = self.get_current_pose()
        rot = current_pose.get_rotation("euler_angles", "ZYX").squeeze().cpu().numpy()
        trans = current_pose.get_translation().squeeze().cpu().numpy()
        print("绝对位姿:")
        print(f"旋转 - alpha: {rot[0]:.4f}, beta: {rot[1]:.4f}, gamma: {rot[2]:.4f} rad")
        print(f"平移 - bx: {trans[0]:.2f}, by: {trans[1]:.2f}, bz: {trans[2]:.2f} mm")
        print("=" * 60)

    def get_current_pose(self):
        """获取当前位姿"""
        rot = torch.tensor([self.pose_params[:3]], dtype=torch.float32, device=self.device)
        xyz = torch.tensor([self.pose_params[3:]], dtype=torch.float32, device=self.device)
        relative_pose = RigidTransform(rot, xyz, parameterization="so3_log_map", device=self.device)
        return relative_pose.compose(self.base_pose)

    def generate_drr_image(self):
        """生成DRR图像"""
        current_pose = self.get_current_pose()
        with torch.no_grad():
            pred_img = self.drr(None, None, None, pose=current_pose, bone_attenuation_multiplier=3)
            pred_img = self.transforms(pred_img).to(self.device).to(torch.float32)
            img_np = pred_img.squeeze().cpu().numpy()
            return img_np

    def update_display(self):
        """更新显示"""
        img_np = self.generate_drr_image()

        # 清除并重新绘制
        self.ax.clear()
        self.ax.imshow(img_np, cmap='gray')
        self.ax.set_title('DRR Image - 实时位姿调整 (按0退出)')
        self.ax.axis('off')

        # 添加参数信息
        param_text = (f"相对调整量:\n"
                      f"alpha: {self.pose_params[0]:.3f} rad\n"
                      f"beta: {self.pose_params[1]:.3f} rad\n"
                      f"gamma: {self.pose_params[2]:.3f} rad\n"
                      f"bx: {self.pose_params[3]:.1f} mm\n"
                      f"by: {self.pose_params[4]:.1f} mm\n"
                      f"bz: {self.pose_params[5]:.1f} mm")

        self.ax.text(0.02, 0.98, param_text, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # 添加控制说明
        control_text = ("键盘控制:\n"
                        "Q/A: alpha +/-\n"
                        "W/S: beta +/-\n"
                        "E/D: gamma +/-\n"
                        "U/J: bx +/-\n"
                        "I/K: by +/-\n"
                        "O/L: bz +/-\n"
                        "R: 重置\n"
                        "0: 退出")

        self.ax.text(0.75, 0.98, control_text, transform=self.ax.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_help(self):
        """显示帮助信息"""
        print("\n🎮 实时键盘控制:")
        print("┌─────────────────┬─────────────────┐")
        print("│    按键         │       功能      │")
        print("├─────────────────┼─────────────────┤")
        print("│      Q          │   alpha +0.05   │")
        print("│      A          │   alpha -0.05   │")
        print("│      W          │   beta +0.05    │")
        print("│      S          │   beta -0.05    │")
        print("│      E          │   gamma +0.05   │")
        print("│      D          │   gamma -0.05   │")
        print("│      U          │   bx +5.0       │")
        print("│      J          │   bx -5.0       │")
        print("│      I          │   by +5.0       │")
        print("│      K          │   by -5.0       │")
        print("│      O          │   bz +5.0       │")
        print("│      L          │   bz -5.0       │")
        print("│      R          │   重置位姿      │")
        print("│      0          │   退出程序      │")
        print("└─────────────────┴─────────────────┘")
        print("💡 提示: 直接按键盘按键即可实时调整，无需回车")

    def run(self):
        """运行交互程序"""
        print("🚀 开始实时位姿调整")
        self.show_help()
        self.print_current_pose()

        # 显示窗口并等待
        plt.show()


def main():
    device = "cuda:0"

    # 初始化DRR
    root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/CT/ChenYuXin/20240530105749/304"
    x_root = "/media/sda1/Data/ERCP/CT+X+MRCP/liwei/陈羽馨/ERCP/YUXING^CHEN^/20240531155445/1"
    specimen = IntubationDataset(root, x_root, z_offset=70)

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
        bone_attenuation_multiplier=3,
    ).to(device)

    transforms = Transforms(height)

    # 显示参考图像
    img, pose = specimen[33]
    img = transforms(img, reverse=False).to(device).to(torch.float32)
    plt.figure(figsize=(10, 8))
    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title("参考X光图像")
    plt.axis('off')
    plt.show()

    # 创建调整器
    adjuster = InteractivePoseAdjuster(drr, transforms, device)

    # 运行交互式调整
    adjuster.run()


if __name__ == "__main__":
    main()