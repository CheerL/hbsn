import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from utils.geodesicwelding import geodesicwelding
from utils.move_image import move_image


class ImageGenerator:
    def __init__(self, h: int=256, w: int=256):
        self.h = h
        self.w = w
    
    def bound2image(self, z: np.ndarray):
        z = z - z.mean()
        z = z / np.abs(z).max()

        fig = plt.figure(figsize=(self.w, self.h), facecolor='black', dpi=1)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('off')
        plt.fill(z.real, z.imag, color='white')
        plt.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(self.h, self.w, 3)
        plt.close(fig)
        return img.copy()
    
    def save_image(self, img: np.ndarray, path: str):
        plt.imsave(path, img)

    def distort_grid(self, distortion_scale=0.03, n=20, m=20):
        """
        INPUT:
            size: int
                Size of the grid (size x size)
            distortion_scale: float
                Scale of the distortion
        OUTPUT:
            grid: size x size x 2 array
                Distorted grid
        """
        x_sparse = np.linspace(-1, 1, n)
        y_sparse = np.linspace(-1, 1, m)
        
        xv, yv = np.meshgrid(x_sparse, y_sparse)
        distortion = distortion_scale * np.random.randn(n, m, 2)
        grid = np.stack([xv, yv], axis=-1) + distortion
        

        grid_x = grid[:, :, 0]
        grid_y = grid[:, :, 1]

        x = np.linspace(-1, 1, self.w)
        y = np.linspace(-1, 1, self.h)
        y, x = np.meshgrid(x, y)
        # 使用双线性插值
        fx = interpolate.RegularGridInterpolator((x_sparse, y_sparse), grid_x)
        fy = interpolate.RegularGridInterpolator((x_sparse, y_sparse), grid_y)

        # 计算插值结果
        grid_dense = np.stack([fx((x, y)), fy((x, y))], axis=-1)
        return grid_dense
    
    def distort_image(self, I, grid):
        """
        INPUT:
            I: H x W x C tensor
                I is the input image
            grid: H x W x 2 tensor / float / tuple
                when tensor, grid is the map, grid(J) = I
                when float, grid is the distortion scale
                when tuple, grid is the distortion scale and other parameters
                
        OUTPUT:
            J: H x W x C tensor
                J is the output image
        """
        if I.ndim == 2:
            C = 1
        else:
            C = I.shape[2]
            
        if isinstance(grid, np.ndarray):
            assert grid.shape == (self.h, self.w, 2)
        elif isinstance(grid, float):
            grid = self.distort_grid(grid)
        elif isinstance(grid, tuple) and (len(grid) == 3 or len(grid) == 1):
            grid = self.distort_grid(**grid)
        
        J = move_image(I, grid, version='scipy').astype(np.uint8)
        # I = torch.DoubleTensor(I).reshape(1, self.h, self.w, C).permute(0, 3, 1, 2)
        # grid = torch.tensor(grid).reshape(1, self.h, self.w, 2)
        # J = F.grid_sample(
        #     I, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        # )
        # J = J.permute(0, 2, 3, 1).reshape(self.h, self.w, C).numpy().astype(np.uint8)
        return J
    
    def generate_image(self):
        raise NotImplementedError

class ConformalWeldingImageGenerator(ImageGenerator):
    def __init__(self, h:int=256, w:int=256, n: int=500):
        super().__init__(h, w)
        self.n = n
        self.x = self.regular_x()
        
    def generate_image(self, k, scale):
        y = self.random_y(k, scale)
        z, _ = geodesicwelding(y, [], y, self.x)
        img = self.bound2image(z)
        return img

    def regular_x(self):
        x = np.arange(self.n)
        x = self.map_to_unit_circle(x)
        return x

    def random_y(self, k, scale):
        gap = np.sort(np.random.randint(1, self.n-1, size=k-1))
        gap = np.insert(gap, 0, 0)
        gap = np.append(gap, self.n-1)
        
        means = np.random.randn(k) * scale
        stds = np.random.randn(k)
        length = np.diff(gap)
        
        y = np.zeros(self.n)
        for i in range(k):
            rd = np.random.randn(length[i])
            rd = rd * stds[i] + means[i]
            y[gap[i]:gap[i+1]] = rd

        y = self.map_to_unit_circle(y)
        return y

    def map_to_unit_circle(self, x: np.ndarray):
        x.sort()
        x = (x - x[0]) / (x[-1] - x[-2] + x[-1] - x[0])
        x = np.exp(1j * 2 * np.pi * x)
        return x
        
class PolygonImageGenerator(ImageGenerator):
    def generate_image(self, n):
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = x + y * 1j
        z = z - z.mean()
        idx = np.angle(z).argsort()
        z = z[idx]
        img = self.bound2image(z)
        return img