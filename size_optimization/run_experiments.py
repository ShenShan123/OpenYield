"""
总控实验文件 - 电路优化算法对比实验
Main Experiment Controller - Circuit Optimization Algorithm Comparison

此文件用于运行和对比不同的优化算法
This file is used to run and compare different optimization algorithms
"""

import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup paths
# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import algorithm modules
# 导入算法模块
try:
    from size_optimization.demo_pso import main as pso_main
    from size_optimization.demo_sa import main as sa_main  
    from size_optimization.demo_cbo import main as cbo_main
    from size_optimization.demo_smac import main as smac_main
    from size_optimization.demo_roseopt import main as rose_opt_main 
except ImportError as e:
    print(f"导入算法模块失败: {e}")
    print("请确保所有算法文件都在正确的位置")
    sys.exit(1)

# Import utilities
# 导入工具函数
from size_optimization.exp_utils import seed_set, create_directories


class ExperimentRunner:
    """
    实验运行器类
    Experiment runner class
    """
    
    def __init__(self, config_path="config_sram.yaml"):
        """
        初始化实验运行器
        Initialize experiment runner
        """
        self.config_path = config_path
        self.results = {}
        self.timing_results = {}
        
        # 确保结果目录存在
        # Ensure results directory exists
        create_directories()
        Path("sim/opt/experiments").mkdir(exist_ok=True, parents=True)
        
        # 可用的算法映射
        # Available algorithm mapping
        self.algorithms = {
            'PSO': {
                'name': 'Particle Swarm Optimization',
                'function': pso_main,
                'description': '粒子群优化算法'
            },
            'SA': {
                'name': 'Simulated Annealing',
                'function': sa_main,
                'description': '模拟退火算法'
            },
            'CBO': {
                'name': 'Constrained Bayesian Optimization',
                'function': cbo_main,
                'description': '约束贝叶斯优化算法'
            },
            'SMAC': {
                'name': 'Sequential Model-based Algorithm Configuration',
                'function': smac_main,
                'description': 'SMAC算法'
            },
            'RoSE_Opt': {
                'name': 'RoSE-Opt (BO + RL)',
                'function': rose_opt_main,
                'description': 'RoSE-Opt算法'
            }
        }
    
    def run_single_algorithm(self, algorithm_name, seed=1):
        """
        运行单个算法
        Run a single algorithm
        
        Args:
            algorithm_name: 算法名称 / Algorithm name
            seed: 随机种子 / Random seed
            
        Returns:
            结果字典 / Result dictionary
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"未知算法: {algorithm_name}. 可用算法: {list(self.algorithms.keys())}")
        
        print("=" * 80)
        print(f"运行算法: {algorithm_name} - {self.algorithms[algorithm_name]['name']}")
        print(f"描述: {self.algorithms[algorithm_name]['description']}")
        print(f"配置文件: {self.config_path}")
        print(f"随机种子: {seed}")
        print("=" * 80)
        
        # 设置随机种子
        # Set random seed
        seed_set(seed)
        
        # 记录开始时间
        # Record start time
        start_time = time.time()
        
        try:
            # 运行算法
            # Run algorithm
            algorithm_function = self.algorithms[algorithm_name]['function']
            result = algorithm_function(self.config_path)
            
            # 记录结束时间
            # Record end time
            end_time = time.time()
            total_time = end_time - start_time
            
            # 保存结果
            # Save results
            self.results[algorithm_name] = result
            self.timing_results[algorithm_name] = total_time
            
            print(f"\n{algorithm_name} 算法完成！")
            print(f"总用时: {total_time:.2f} 秒")
            
            if result and result.get('params') is not None:
                print(f"最佳Merit: {result.get('merit', 'N/A'):.6e}")
                print(f"迭代次数: {result.get('iteration', 'N/A')}")
            else:
                print("未找到有效解")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n{algorithm_name} 算法运行失败！")
            print(f"错误信息: {str(e)}")
            print(f"用时: {total_time:.2f} 秒")
            
            # 保存失败结果
            # Save failure result
            failed_result = {
                'params': None,
                'merit': None,
                'result': None,
                'iteration': -1,
                'error': str(e)
            }
            
            self.results[algorithm_name] = failed_result
            self.timing_results[algorithm_name] = total_time
            
            return failed_result
    
    def run_multiple_algorithms(self, algorithm_list, seed=1):
        """
        运行多个算法
        Run multiple algorithms
        
        Args:
            algorithm_list: 算法列表 / List of algorithms
            seed: 随机种子 / Random seed
        """
        print("=" * 80)
        print("开始多算法对比实验")
        print(f"算法列表: {algorithm_list}")
        print(f"配置文件: {self.config_path}")
        print(f"随机种子: {seed}")
        print("=" * 80)
        
        for algorithm in algorithm_list:
            print(f"\n正在运行 {algorithm}...")
            self.run_single_algorithm(algorithm, seed)
            print(f"{algorithm} 完成\n")
        
        # 生成对比报告
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """
        生成算法对比报告
        Generate algorithm comparison report
        """
        print("=" * 80)
        print("算法对比报告 / Algorithm Comparison Report")
        print("=" * 80)
        
        # 创建结果表格
        # Create results table
        comparison_data = []
        
        for algo_name, result in self.results.items():
            row = {
                'Algorithm': algo_name,
                'Algorithm_Name': self.algorithms[algo_name]['name'],
                'Success': result.get('params') is not None,
                'Best_Merit': result.get('merit', None),
                'Iterations': result.get('iteration', None),
                'Time_seconds': self.timing_results.get(algo_name, None),
                'Error': result.get('error', None)
            }
            comparison_data.append(row)
        
        # 转换为DataFrame
        # Convert to DataFrame
        df = pd.DataFrame(comparison_data)
        
        # 打印表格
        # Print table
        print("\n结果汇总:")
        print(df.to_string(index=False))
        
        # 保存详细结果
        # Save detailed results
        self.save_detailed_results(df)
        
        # 生成图表
        # Generate plots
        self.generate_plots(df)
        
        # 输出最佳结果
        # Output best results
        self.print_best_results(df)
    
    def save_detailed_results(self, summary_df):
        """
        保存详细结果
        Save detailed results
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总结果
        # Save summary results
        summary_file = f"sim/opt/experiments/summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存到: {summary_file}")
        
        # 保存详细结果
        # Save detailed results
        detailed_file = f"sim/opt/experiments/detailed_{timestamp}.json"
        detailed_results = {
            'config_file': self.config_path,
            'timestamp': timestamp,
            'results': self.results,
            'timing': self.timing_results
        }
        
        # 自定义JSON编码器处理特殊类型
        # Custom JSON encoder for special types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):  # objects
                    return str(obj)
                else:
                    return super().default(obj)
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
        
        print(f"详细结果已保存到: {detailed_file}")
    
    def generate_plots(self, df):
        """
        生成对比图表
        Generate comparison plots
        """
        try:
            # 创建图表目录
            # Create plots directory
            plots_dir = Path("sim/opt/experiments/plots")
            plots_dir.mkdir(exist_ok=True, parents=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 筛选成功的结果
            # Filter successful results
            successful_df = df[df['Success'] == True].copy()
            
            if len(successful_df) == 0:
                print("没有成功的结果可用于绘图")
                return
            
            # 1. Merit对比图
            # 1. Merit comparison plot
            plt.figure(figsize=(12, 6))  # wider figure for more algorithms
            algorithms = successful_df['Algorithm'].tolist()
            merits = successful_df['Best_Merit'].tolist()
            
            # more colors for 5 algorithms
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            bars = plt.bar(algorithms, merits, color=colors[:len(algorithms)])
            plt.xlabel('算法 / Algorithm')
            plt.ylabel('最佳Merit值 / Best Merit Value')
            plt.title('算法Merit值对比 / Algorithm Merit Comparison')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            # Add value labels
            for bar, merit in zip(bars, merits):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{merit:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"merit_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 运行时间对比图
            # 2. Runtime comparison plot
            plt.figure(figsize=(12, 6))  # wider figure
            times = successful_df['Time_seconds'].tolist()
            
            bars = plt.bar(algorithms, times, color=colors[:len(algorithms)])
            plt.xlabel('算法 / Algorithm')
            plt.ylabel('运行时间 (秒) / Runtime (seconds)')
            plt.title('算法运行时间对比 / Algorithm Runtime Comparison')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            # Add value labels
            for bar, time_val in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01, 
                        f'{time_val:.1f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"runtime_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 迭代次数对比图
            # 3. Iterations comparison plot
            plt.figure(figsize=(12, 6))  # wider figure
            iterations = successful_df['Iterations'].tolist()
            
            bars = plt.bar(algorithms, iterations, color=colors[:len(algorithms)])
            plt.xlabel('算法 / Algorithm')
            plt.ylabel('迭代次数 / Iterations')
            plt.title('算法迭代次数对比 / Algorithm Iterations Comparison')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            # Add value labels
            for bar, iter_val in zip(bars, iterations):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iterations)*0.01, 
                        f'{iter_val}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"iterations_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"对比图表已保存到: {plots_dir}")
            
        except Exception as e:
            print(f"生成图表时出错: {e}")
    
    def print_best_results(self, df):
        """
        输出最佳结果
        Print best results
        """
        successful_df = df[df['Success'] == True]
        
        if len(successful_df) == 0:
            print("\n没有成功的优化结果")
            return
        
        print("\n" + "=" * 80)
        print("最佳结果分析 / Best Results Analysis")
        print("=" * 80)
        
        # 最佳Merit
        # Best Merit
        best_merit_idx = successful_df['Best_Merit'].idxmax()
        best_merit_algo = successful_df.loc[best_merit_idx]
        print(f"\n 最佳Merit算法: {best_merit_algo['Algorithm']} ({best_merit_algo['Algorithm_Name']})")
        print(f"   Merit值: {best_merit_algo['Best_Merit']:.6e}")
        print(f"   迭代次数: {best_merit_algo['Iterations']}")
        print(f"   运行时间: {best_merit_algo['Time_seconds']:.2f} 秒")
        
        # 最快算法
        # Fastest algorithm
        fastest_idx = successful_df['Time_seconds'].idxmin()
        fastest_algo = successful_df.loc[fastest_idx]
        print(f"\n 最快算法: {fastest_algo['Algorithm']} ({fastest_algo['Algorithm_Name']})")
        print(f"   运行时间: {fastest_algo['Time_seconds']:.2f} 秒")
        print(f"   Merit值: {fastest_algo['Best_Merit']:.6e}")
        
        # 最少迭代
        # Fewest iterations
        min_iter_idx = successful_df['Iterations'].idxmin()
        min_iter_algo = successful_df.loc[min_iter_idx]
        print(f"\n 最少迭代算法: {min_iter_algo['Algorithm']} ({min_iter_algo['Algorithm_Name']})")
        print(f"   迭代次数: {min_iter_algo['Iterations']}")
        print(f"   Merit值: {min_iter_algo['Best_Merit']:.6e}")
        
        print("\n" + "=" * 80)


def main():
    """
    主函数 - 配置和运行实验
    Main function - configure and run experiments
    """
    print("=" * 80)
    print("电路优化算法对比实验系统")
    print("Circuit Optimization Algorithm Comparison System")
    print("=" * 80)
    
    # ====================================================================
    # 实验配置区域 - 在这里修改实验设置
    # Experiment Configuration - Modify experiment settings here
    # ====================================================================
    
    # 配置文件路径
    # Configuration file path
    CONFIG_FILE = os.path.join(current_dir, "config_sram.yaml")
    
    # 要运行的算法列表 - 在这里选择要对比的算法
    # Algorithm list to run - Select algorithms to compare here
    ALGORITHMS_TO_RUN = [
        'PSO',        # 粒子群优化
        'SA',         # 模拟退火
        'CBO',        # 约束贝叶斯优化
        'SMAC',       # SMAC算法
        'RoSE_Opt'    # RoSE-Opt优化
    ]
    
    # 如果只想运行单个算法，取消下面某行的注释:
    # If you want to run only one algorithm, uncomment one of the lines below:
    # ALGORITHMS_TO_RUN = ['PSO']        # 只运行PSO
    # ALGORITHMS_TO_RUN = ['SA']         # 只运行SA
    # ALGORITHMS_TO_RUN = ['CBO']        # 只运行CBO
    # ALGORITHMS_TO_RUN = ['SMAC']       # 只运行SMAC
    # ALGORITHMS_TO_RUN = ['RoSE_Opt']   # 只运行RoSE-Opt
    
    # 随机种子
    # Random seed
    RANDOM_SEED = 1
    
    # ====================================================================
    # 实验执行区域 - 一般不需要修改
    # Experiment Execution - Usually no need to modify
    # ====================================================================
    
    # 检查配置文件是否存在
    # Check if configuration file exists
    if not Path(CONFIG_FILE).exists():
        print(f"错误: 配置文件 {CONFIG_FILE} 不存在！")
        print("请确保配置文件在正确的位置")
        return
    
    # 创建实验运行器
    # Create experiment runner
    runner = ExperimentRunner(CONFIG_FILE)
    
    # 显示实验配置
    # Display experiment configuration
    print(f"配置文件: {CONFIG_FILE}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"算法列表: {ALGORITHMS_TO_RUN}")
    print("\n可用算法:")
    for algo, info in runner.algorithms.items():
        status = "✓" if algo in ALGORITHMS_TO_RUN else " "
        print(f"  [{status}] {algo}: {info['description']}")
    
    # 确认运行
    # Confirm execution
    print(f"\n即将运行 {len(ALGORITHMS_TO_RUN)} 个算法...")
    input("按回车键开始实验，或 Ctrl+C 取消...")
    
    try:
        # 运行实验
        # Run experiments
        runner.run_multiple_algorithms(ALGORITHMS_TO_RUN, RANDOM_SEED)
        
        print("\n" + "=" * 80)
        print("所有实验完成！结果已保存到 sim/opt/experiments/ 目录")
        print("All experiments completed! Results saved to sim/opt/experiments/ directory")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        print("Experiments interrupted by user")
    except Exception as e:
        print(f"\n实验运行过程中发生错误: {e}")
        print(f"Error occurred during experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
