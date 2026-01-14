"""
生成复杂曲线数据集
用于测试 MLP 网络的拟合能力
"""
import numpy as np
import utils


def generate_multi_sine(x, noise_std=0.1):
    """多频率正弦叠加（Fourier风格）
    y = sin(x) + 0.5*sin(3x) + 0.3*sin(5x) + 0.2*sin(7x) + noise
    """
    y = (np.sin(x) + 
         0.5 * np.sin(3 * x) + 
         0.3 * np.sin(5 * x) + 
         0.2 * np.sin(7 * x))
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_compound_trig(x, noise_std=0.1):
    """复合三角函数
    y = sin(x)*cos(2x) + sin(3x)*cos(x) + noise
    """
    y = np.sin(x) * np.cos(2 * x) + np.sin(3 * x) * np.cos(x)
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_high_order_poly(x, noise_std=0.1):
    """高阶多项式
    y = 0.1*x^5 - 0.5*x^3 + x + noise
    """
    y = 0.1 * x**5 - 0.5 * x**3 + x
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_mixed_transcendental(x, noise_std=0.1):
    """混合超越函数
    y = tanh(2x)*sin(3x) + exp(-x^2)*cos(5x) + noise
    """
    y = np.tanh(2 * x) * np.sin(3 * x) + np.exp(-x**2) * np.cos(5 * x)
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_piecewise(x, noise_std=0.1):
    """分段非线性函数
    不同区间使用不同函数，创造复杂的分段曲线
    """
    y = np.zeros_like(x)
    
    # 区间1: x < -3, 使用 sin(2x) + x
    mask1 = x < -3
    y[mask1] = np.sin(2 * x[mask1]) + x[mask1]
    
    # 区间2: -3 <= x < -1, 使用 exp(x) * cos(3x)
    mask2 = (x >= -3) & (x < -1)
    y[mask2] = np.exp(x[mask2]) * np.cos(3 * x[mask2])
    
    # 区间3: -1 <= x < 1, 使用 tanh(3x) * 2
    mask3 = (x >= -1) & (x < 1)
    y[mask3] = np.tanh(3 * x[mask3]) * 2
    
    # 区间4: 1 <= x < 3, 使用 sin(x^2)
    mask4 = (x >= 1) & (x < 3)
    y[mask4] = np.sin(x[mask4]**2)
    
    # 区间5: x >= 3, 使用 log(x) * sin(5x)
    mask5 = x >= 3
    y[mask5] = np.log(x[mask5]) * np.sin(5 * x[mask5])
    
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_extreme_complex(x, noise_std=0.1):
    """极限复杂曲线（推荐）
    组合多个波形、多项式、指数和三角函数
    y = sin(x)*sin(2x)*sin(3x) + 0.5*tanh(x) + 0.3*exp(-x^2/2)*cos(4x) + 0.1*x^2 + noise
    """
    y = (np.sin(x) * np.sin(2 * x) * np.sin(3 * x) + 
         0.5 * np.tanh(x) + 
         0.3 * np.exp(-x**2 / 2) * np.cos(4 * x) + 
         0.1 * x**2)
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


def generate_ultra_complex(x, noise_std=0.1):
    """超级复杂曲线
    结合更多频率和非线性变换，创造最难拟合的平滑曲线
    """
    y = (
        # 多频率正弦叠加（主要成分）
        np.sin(x) * np.sin(2*x) * np.sin(3*x) +
        # 高频振荡（中心区域更明显）
        0.3 * np.sin(10 * x) * np.exp(-0.3 * x**2) +
        # 双曲正切的振荡
        0.4 * np.tanh(2 * x) * np.cos(5 * x) +
        # 指数衰减的复杂波形
        0.2 * np.exp(-np.abs(x) / 3) * np.sin(8 * x) +
        # 多项式项（整体趋势）
        0.03 * x**3 - 0.1 * x +
        # 额外的平滑非线性项（替代sign函数）
        0.3 * np.sin(4 * x) * np.cos(3 * x) +
        # 更多高频成分
        0.15 * np.sin(12 * x) * np.exp(-0.5 * x**2)
    )
    noise = np.random.randn(*x.shape) * noise_std
    return y + noise


# 所有可用的生成函数
GENERATORS = {
    'multi_sine': ('多频率正弦叠加', generate_multi_sine),
    'compound_trig': ('复合三角函数', generate_compound_trig),
    'high_poly': ('高阶多项式', generate_high_order_poly),
    'mixed_trans': ('混合超越函数', generate_mixed_transcendental),
    'piecewise': ('分段非线性', generate_piecewise),
    'extreme': ('极限复杂曲线', generate_extreme_complex),
    'ultra': ('超级复杂曲线', generate_ultra_complex),
}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成复杂曲线数据集')
    parser.add_argument('--type', '-t', type=str, default='ultra',
                        choices=list(GENERATORS.keys()),
                        help='曲线类型 (默认: ultra)')
    parser.add_argument('--num', '-n', type=int, default=1000,
                        help='数据点数量 (默认: 1000)')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='噪声标准差 (默认: 0.1)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出文件名 (默认: data_complex.csv)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='显示数据散点图')
    
    args = parser.parse_args()
    
    # 生成 X 数据
    X = np.linspace(-5, 5, args.num).reshape(-1, 1)
    
    # 获取生成函数
    name, generator = GENERATORS[args.type]
    print(f'生成曲线类型: {name}')
    
    # 生成 Y 数据
    Y = generator(X.flatten(), noise_std=args.noise).reshape(-1, 1)
    
    # 确定输出文件名
    output_file = args.output if args.output else f'data_{args.type}.csv'
    
    # 保存数据
    utils.save_to_csv(X, Y, output_file)
    
    # 显示散点图
    if args.show:
        utils.draw_2d_scatter(X, Y)


if __name__ == '__main__':
    main()
