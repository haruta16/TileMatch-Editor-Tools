#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验模式配置[1,2,3]对TileMatch指标影响的全面分析工具
单文件实现：数据加载、相关性分析、机器学习建模、可视化展示

作者: Claude Code Assistant
创建时间: 2024-09-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class ExperienceConfigAnalyzer:
    """体验模式配置分析器 - 单类集成所有分析功能"""

    def __init__(self, csv_path: str = None):
        """初始化分析器

        Args:
            csv_path: CSV数据文件路径，如为None则自动检测
        """
        self.csv_path = csv_path or self._find_latest_csv()
        self.data = None
        self.features = None
        self.target_metrics = [
            'DifficultyScore', 'PeakDockCount', 'PressureValueMean',
            'PressureValueMax', 'PressureValueStdDev', 'FinalDifficulty',
            'TotalMoves', 'InitialMinCost', 'DifficultyPosition'
        ]
        self.results = {}

    def _find_latest_csv(self) -> str:
        """自动查找最新的BattleAnalysis CSV文件"""
        current_dir = Path(__file__).parent
        csv_files = list(current_dir.glob("BattleAnalysis*.csv"))

        if not csv_files:
            # 查找BattleAnalysisResults子目录
            results_dir = current_dir / "BattleAnalysisResults"
            if results_dir.exists():
                csv_files = list(results_dir.glob("BattleAnalysis*.csv"))

        if not csv_files:
            raise FileNotFoundError("未找到BattleAnalysis CSV文件，请确保数据文件存在")

        # 返回最新的文件
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"🔍 自动检测到CSV文件: {latest_file}")
        return str(latest_file)

    def load_and_preprocess(self) -> 'ExperienceConfigAnalyzer':
        """加载数据并进行预处理"""
        print("📊 加载和预处理数据...")

        # 加载CSV数据
        self.data = pd.read_csv(self.csv_path, encoding='utf-8')
        print(f"✅ 数据加载完成: {len(self.data)}条记录")

        # 解析体验模式配置
        self._parse_experience_config()

        # 生成特征工程
        self._create_features()

        # 数据清洗
        self._clean_data()

        print(f"📈 预处理完成: {len(self.data)}条有效记录, {len(self.features.columns)}个特征")
        return self

    def _parse_experience_config(self):
        """解析体验模式配置[1,2,3]格式"""
        def parse_config(config_str):
            """解析配置字符串为位置数值"""
            try:
                # 移除引号和括号，分割数字
                clean_str = str(config_str).strip('[]"()')
                numbers = [int(x.strip()) for x in clean_str.split(',')]
                return numbers[:3] if len(numbers) >= 3 else [1, 2, 3]  # 默认值
            except:
                return [1, 2, 3]  # 错误时返回默认值

        # 解析配置
        configs = self.data['ExperienceMode'].apply(parse_config)

        # 提取位置特征
        self.data['pos1'] = configs.apply(lambda x: x[0])
        self.data['pos2'] = configs.apply(lambda x: x[1])
        self.data['pos3'] = configs.apply(lambda x: x[2])

    def _create_features(self):
        """创建特征工程"""
        # 基础位置特征
        features_df = self.data[['pos1', 'pos2', 'pos3']].copy()

        # 统计特征
        features_df['config_mean'] = (self.data['pos1'] + self.data['pos2'] + self.data['pos3']) / 3
        features_df['config_std'] = np.sqrt(((self.data['pos1'] - features_df['config_mean'])**2 +
                                            (self.data['pos2'] - features_df['config_mean'])**2 +
                                            (self.data['pos3'] - features_df['config_mean'])**2) / 3)
        features_df['config_range'] = np.maximum.reduce([self.data['pos1'], self.data['pos2'], self.data['pos3']]) - \
                                     np.minimum.reduce([self.data['pos1'], self.data['pos2'], self.data['pos3']])
        features_df['config_sum'] = self.data['pos1'] + self.data['pos2'] + self.data['pos3']

        # 交互特征
        features_df['pos1_pos2'] = self.data['pos1'] * self.data['pos2']
        features_df['pos1_pos3'] = self.data['pos1'] * self.data['pos3']
        features_df['pos2_pos3'] = self.data['pos2'] * self.data['pos3']
        features_df['pos_product'] = self.data['pos1'] * self.data['pos2'] * self.data['pos3']

        # 序列模式特征
        features_df['is_increasing'] = ((self.data['pos1'] <= self.data['pos2']) &
                                      (self.data['pos2'] <= self.data['pos3'])).astype(int)
        features_df['is_decreasing'] = ((self.data['pos1'] >= self.data['pos2']) &
                                      (self.data['pos2'] >= self.data['pos3'])).astype(int)
        features_df['is_uniform'] = ((self.data['pos1'] == self.data['pos2']) &
                                   (self.data['pos2'] == self.data['pos3'])).astype(int)

        # 极值特征
        features_df['has_extreme_low'] = ((self.data['pos1'] == 1) | (self.data['pos2'] == 1) | (self.data['pos3'] == 1)).astype(int)
        features_df['has_extreme_high'] = ((self.data['pos1'] == 9) | (self.data['pos2'] == 9) | (self.data['pos3'] == 9)).astype(int)

        self.features = features_df

    def _clean_data(self):
        """数据清洗"""
        # 移除GameCompleted=False的记录（游戏未完成）
        initial_count = len(self.data)
        self.data = self.data[self.data['GameCompleted'] == True].copy()

        # 移除目标指标为空值的记录
        for metric in self.target_metrics:
            if metric in self.data.columns:
                self.data = self.data[self.data[metric].notna()].copy()

        # 重新索引特征数据
        self.features = self.features.loc[self.data.index].copy()

        print(f"🧹 数据清洗: {initial_count} -> {len(self.data)}条记录")

    def correlation_analysis(self) -> Dict:
        """相关性分析"""
        print("🔗 执行相关性分析...")

        correlations = {}

        # 位置特征与指标的相关性
        position_corrs = {}
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_corrs = {}
            for metric in self.target_metrics:
                if metric in self.data.columns:
                    corr, p_value = pearsonr(self.features[pos], self.data[metric])
                    pos_corrs[metric] = {'correlation': corr, 'p_value': p_value, 'significant': p_value < 0.05}
            position_corrs[pos] = pos_corrs

        correlations['position_correlations'] = position_corrs

        # 特征重要性（基于所有特征）
        feature_importance = {}
        for metric in self.target_metrics:
            if metric in self.data.columns:
                # 使用随机森林计算特征重要性
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(self.features, self.data[metric])

                importance_dict = dict(zip(self.features.columns, rf.feature_importances_))
                feature_importance[metric] = dict(sorted(importance_dict.items(),
                                                       key=lambda x: x[1], reverse=True))

        correlations['feature_importance'] = feature_importance

        self.results['correlations'] = correlations
        print("✅ 相关性分析完成")
        return correlations

    def pattern_analysis(self) -> Dict:
        """配置模式分析"""
        print("🎯 执行配置模式分析...")

        patterns = {}

        # 序列模式分析
        sequence_patterns = {
            'increasing': self.data[self.features['is_increasing'] == 1],
            'decreasing': self.data[self.features['is_decreasing'] == 1],
            'uniform': self.data[self.features['is_uniform'] == 1],
            'mixed': self.data[(self.features['is_increasing'] == 0) &
                             (self.features['is_decreasing'] == 0) &
                             (self.features['is_uniform'] == 0)]
        }

        pattern_stats = {}
        for pattern_name, pattern_data in sequence_patterns.items():
            if len(pattern_data) > 0:
                stats_dict = {}
                for metric in self.target_metrics:
                    if metric in pattern_data.columns:
                        stats_dict[metric] = {
                            'mean': pattern_data[metric].mean(),
                            'std': pattern_data[metric].std(),
                            'count': len(pattern_data)
                        }
                pattern_stats[pattern_name] = stats_dict

        patterns['sequence_patterns'] = pattern_stats

        # 数值分布分析
        value_analysis = {}
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_analysis = {}
            for value in range(1, 10):
                value_data = self.data[self.features[pos] == value]
                if len(value_data) > 0:
                    value_stats = {}
                    for metric in self.target_metrics:
                        if metric in value_data.columns:
                            value_stats[metric] = {
                                'mean': value_data[metric].mean(),
                                'count': len(value_data)
                            }
                    pos_analysis[f'value_{value}'] = value_stats
            value_analysis[pos] = pos_analysis

        patterns['value_distribution'] = value_analysis

        self.results['patterns'] = patterns
        print("✅ 配置模式分析完成")
        return patterns

    def build_prediction_models(self) -> Dict:
        """构建预测模型"""
        print("🤖 构建预测模型...")

        models = {}

        for metric in self.target_metrics:
            if metric not in self.data.columns:
                continue

            # 数据准备
            X = self.features
            y = self.data[metric]

            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 随机森林模型
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # 预测和评估
            y_pred = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            models[metric] = {
                'model': rf,
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': dict(zip(X.columns, rf.feature_importances_))
            }

        self.results['models'] = models
        print(f"✅ 预测模型构建完成，平均R²: {np.mean([m['r2_score'] for m in models.values()]):.3f}")
        return models

    def create_visualizations(self, output_dir: str = None):
        """创建可视化图表"""
        print("📊 创建可视化图表...")

        if output_dir is None:
            output_dir = Path(self.csv_path).parent / "analysis_charts"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. 位置相关性热力图
        self._plot_position_correlation_heatmap(output_path)

        # 2. 配置模式箱线图
        self._plot_pattern_boxplots(output_path)

        # 3. 特征重要性图
        self._plot_feature_importance(output_path)

        # 4. 配置分布散点图
        self._plot_config_distribution(output_path)

        # 5. 模型性能图
        self._plot_model_performance(output_path)

        print(f"✅ 可视化图表已保存到: {output_path}")

    def _plot_position_correlation_heatmap(self, output_path: Path):
        """绘制位置相关性热力图"""
        if 'correlations' not in self.results:
            return

        # 准备数据
        pos_corrs = self.results['correlations']['position_correlations']

        corr_matrix = []
        for pos in ['pos1', 'pos2', 'pos3']:
            row = []
            for metric in self.target_metrics:
                if metric in pos_corrs[pos]:
                    row.append(pos_corrs[pos][metric]['correlation'])
                else:
                    row.append(0)
            corr_matrix.append(row)

        corr_df = pd.DataFrame(corr_matrix,
                              index=['位置1', '位置2', '位置3'],
                              columns=self.target_metrics)

        # 绘图
        plt.figure(figsize=(12, 6))
        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={'label': '相关系数'})
        plt.title('体验配置位置与指标相关性热力图', fontsize=14, fontweight='bold')
        plt.xlabel('性能指标')
        plt.ylabel('配置位置')
        plt.tight_layout()
        plt.savefig(output_path / 'position_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pattern_boxplots(self, output_path: Path):
        """绘制配置模式箱线图"""
        if 'patterns' not in self.results:
            return

        # 为关键指标创建箱线图
        key_metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 序列模式对比
        pattern_data = []
        for pattern in ['increasing', 'decreasing', 'uniform', 'mixed']:
            mask = getattr(self.features, f'is_{pattern}', None)
            if pattern == 'mixed':
                mask = ((self.features['is_increasing'] == 0) &
                       (self.features['is_decreasing'] == 0) &
                       (self.features['is_uniform'] == 0))
            else:
                mask = self.features[f'is_{pattern}'] == 1

            if mask.any():
                pattern_subset = self.data[mask].copy()
                pattern_subset['pattern'] = pattern
                pattern_data.append(pattern_subset)

        if pattern_data:
            combined_data = pd.concat(pattern_data, ignore_index=True)

            for i, metric in enumerate(key_metrics[:3]):
                if metric in combined_data.columns:
                    sns.boxplot(data=combined_data, x='pattern', y=metric, ax=axes[i])
                    axes[i].set_title(f'{metric} - 序列模式对比')
                    axes[i].tick_params(axis='x', rotation=45)

        # 位置数值分布
        position_data = []
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_subset = self.data.copy()
            pos_subset['position'] = pos
            pos_subset['value'] = self.features[pos]
            position_data.append(pos_subset)

        if position_data:
            pos_combined = pd.concat(position_data, ignore_index=True)
            sns.boxplot(data=pos_combined, x='value', y='DifficultyScore',
                       hue='position', ax=axes[3])
            axes[3].set_title('DifficultyScore - 位置数值分布')
            axes[3].legend(title='位置')

        plt.suptitle('体验配置模式分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'pattern_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, output_path: Path):
        """绘制特征重要性图"""
        if 'models' not in self.results:
            return

        # 选择前3个重要指标
        key_metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, metric in enumerate(key_metrics):
            if metric in self.results['models']:
                importance = self.results['models'][metric]['feature_importance']

                # 取前8个重要特征
                top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])

                features = list(top_features.keys())
                values = list(top_features.values())

                bars = axes[i].barh(features, values)
                axes[i].set_title(f'{metric}\n特征重要性 (R²={self.results["models"][metric]["r2_score"]:.3f})')
                axes[i].set_xlabel('重要性分数')

                # 颜色映射
                colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

        plt.suptitle('机器学习模型特征重要性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_config_distribution(self, output_path: Path):
        """绘制配置分布散点图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 3D配置空间投影 (pos1 vs pos2，颜色表示pos3)
        scatter = axes[0,0].scatter(self.features['pos1'], self.features['pos2'],
                                  c=self.features['pos3'], cmap='viridis', alpha=0.6)
        axes[0,0].set_xlabel('位置1数值')
        axes[0,0].set_ylabel('位置2数值')
        axes[0,0].set_title('配置空间分布 (颜色=位置3)')
        plt.colorbar(scatter, ax=axes[0,0], label='位置3数值')

        # DifficultyScore分布
        if 'DifficultyScore' in self.data.columns:
            scatter2 = axes[0,1].scatter(self.features['config_mean'], self.features['config_std'],
                                       c=self.data['DifficultyScore'], cmap='coolwarm', alpha=0.6)
            axes[0,1].set_xlabel('配置均值')
            axes[0,1].set_ylabel('配置标准差')
            axes[0,1].set_title('难度分数分布')
            plt.colorbar(scatter2, ax=axes[0,1], label='DifficultyScore')

        # 序列模式分布
        pattern_colors = {'increasing': 'red', 'decreasing': 'blue', 'uniform': 'green', 'mixed': 'gray'}
        for pattern, color in pattern_colors.items():
            if pattern == 'mixed':
                mask = ((self.features['is_increasing'] == 0) &
                       (self.features['is_decreasing'] == 0) &
                       (self.features['is_uniform'] == 0))
            else:
                mask = self.features[f'is_{pattern}'] == 1

            if mask.any():
                axes[1,0].scatter(self.features[mask]['pos1'], self.features[mask]['pos3'],
                                alpha=0.6, c=color, label=pattern, s=20)

        axes[1,0].set_xlabel('位置1数值')
        axes[1,0].set_ylabel('位置3数值')
        axes[1,0].set_title('序列模式分布')
        axes[1,0].legend()

        # 极值组合分析
        extreme_mask = (self.features['has_extreme_low'] == 1) | (self.features['has_extreme_high'] == 1)
        normal_mask = ~extreme_mask

        if 'PeakDockCount' in self.data.columns:
            axes[1,1].scatter(self.data[normal_mask]['PeakDockCount'],
                            self.data[normal_mask]['DifficultyScore'] if 'DifficultyScore' in self.data.columns else 0,
                            alpha=0.5, c='blue', label='常规配置', s=20)
            axes[1,1].scatter(self.data[extreme_mask]['PeakDockCount'],
                            self.data[extreme_mask]['DifficultyScore'] if 'DifficultyScore' in self.data.columns else 0,
                            alpha=0.7, c='red', label='极值配置', s=20)
            axes[1,1].set_xlabel('PeakDockCount')
            axes[1,1].set_ylabel('DifficultyScore')
            axes[1,1].set_title('极值配置效应分析')
            axes[1,1].legend()

        plt.suptitle('体验配置分布特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'config_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_performance(self, output_path: Path):
        """绘制模型性能图"""
        if 'models' not in self.results:
            return

        metrics = list(self.results['models'].keys())
        r2_scores = [self.results['models'][m]['r2_score'] for m in metrics]
        rmse_scores = [self.results['models'][m]['rmse'] for m in metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # R²分数
        bars1 = ax1.bar(metrics, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title('预测模型R²性能评分')
        ax1.set_ylabel('R²分数')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='基准线(0.5)')
        ax1.legend()

        # 为每个bar添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # RMSE分数 (标准化显示)
        bars2 = ax2.bar(metrics, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('预测模型RMSE误差')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)

        plt.suptitle('机器学习模型性能评估', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_path: str = None) -> str:
        """生成分析报告"""
        print("📝 生成分析报告...")

        if output_path is None:
            output_path = Path(self.csv_path).parent / "analysis_report.md"

        report = []
        report.append("# 体验模式配置[1,2,3]分析报告\n")
        report.append(f"**数据源**: {self.csv_path}")
        report.append(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**样本数量**: {len(self.data):,}条\n")

        # 数据概览
        report.append("## 📊 数据概览\n")
        report.append(f"- 配置范围: pos1=[{self.features['pos1'].min()}-{self.features['pos1'].max()}], " +
                     f"pos2=[{self.features['pos2'].min()}-{self.features['pos2'].max()}], " +
                     f"pos3=[{self.features['pos3'].min()}-{self.features['pos3'].max()}]")
        report.append(f"- 唯一配置数: {len(self.data[['pos1', 'pos2', 'pos3']].drop_duplicates())}种")

        # 相关性分析结果
        if 'correlations' in self.results:
            report.append("\n## 🔗 相关性分析结果\n")

            # 位置重要性排序
            pos_importance = {}
            for pos in ['pos1', 'pos2', 'pos3']:
                avg_abs_corr = np.mean([abs(self.results['correlations']['position_correlations'][pos][m]['correlation'])
                                      for m in self.target_metrics
                                      if m in self.results['correlations']['position_correlations'][pos]])
                pos_importance[pos] = avg_abs_corr

            sorted_positions = sorted(pos_importance.items(), key=lambda x: x[1], reverse=True)

            report.append("### 位置重要性排序:")
            for i, (pos, importance) in enumerate(sorted_positions, 1):
                pos_name = f"位置{pos[-1]}"
                report.append(f"{i}. **{pos_name}**: {importance:.3f} (平均绝对相关系数)")

        # 模式分析结果
        if 'patterns' in self.results:
            report.append("\n## 🎯 配置模式分析\n")

            # 序列模式统计
            seq_patterns = self.results['patterns']['sequence_patterns']
            if seq_patterns:
                report.append("### 序列模式分布:")
                for pattern, stats in seq_patterns.items():
                    count = stats.get('DifficultyScore', {}).get('count', 0)
                    if count > 0:
                        avg_difficulty = stats.get('DifficultyScore', {}).get('mean', 0)
                        pattern_name = {'increasing': '递增', 'decreasing': '递减',
                                      'uniform': '相等', 'mixed': '混合'}[pattern]
                        report.append(f"- **{pattern_name}模式**: {count}个样本, 平均难度: {avg_difficulty:.2f}")

        # 模型预测结果
        if 'models' in self.results:
            report.append("\n## 🤖 预测模型性能\n")

            model_performance = [(metric, model['r2_score']) for metric, model in self.results['models'].items()]
            model_performance.sort(key=lambda x: x[1], reverse=True)

            report.append("### 模型R²性能排序:")
            for i, (metric, r2) in enumerate(model_performance, 1):
                performance_level = "优秀" if r2 > 0.7 else "良好" if r2 > 0.5 else "一般" if r2 > 0.3 else "较差"
                report.append(f"{i}. **{metric}**: R²={r2:.3f} ({performance_level})")

        # 关键发现
        report.append("\n## 💡 关键发现\n")
        report.append("### 主要结论:")

        # 基于分析结果生成结论
        if 'correlations' in self.results and 'models' in self.results:
            # 找出预测性能最好的指标
            best_metric = max(self.results['models'].items(), key=lambda x: x[1]['r2_score'])
            report.append(f"1. **{best_metric[0]}** 是最可预测的指标 (R²={best_metric[1]['r2_score']:.3f})")

            # 找出最重要的位置
            most_important_pos = sorted_positions[0][0] if 'sorted_positions' in locals() else 'pos1'
            report.append(f"2. **{most_important_pos}** 对整体性能影响最大")

            report.append(f"3. 共识别出 {len(self.features.columns)} 个有效特征维度")

        report.append("\n### 优化建议:")
        report.append("1. 重点关注影响力最大的位置配置")
        report.append("2. 考虑序列模式对性能的系统性影响")
        report.append("3. 避免极值配置可能带来的不稳定性")

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ 分析报告已保存到: {output_path}")
        return str(output_path)

    def run_complete_analysis(self, output_dir: str = None) -> Dict:
        """运行完整分析流程"""
        print("🚀 开始完整体验配置分析...")

        # 执行所有分析步骤
        self.load_and_preprocess()
        correlations = self.correlation_analysis()
        patterns = self.pattern_analysis()
        models = self.build_prediction_models()

        # 创建可视化
        self.create_visualizations(output_dir)

        # 生成报告
        report_path = self.generate_report()

        # 汇总结果
        summary = {
            'data_summary': {
                'total_records': len(self.data),
                'unique_configs': len(self.data[['pos1', 'pos2', 'pos3']].drop_duplicates()),
                'target_metrics': len(self.target_metrics)
            },
            'analysis_results': {
                'correlations_completed': len(correlations) > 0,
                'patterns_completed': len(patterns) > 0,
                'models_completed': len(models) > 0,
                'best_model_r2': max([m['r2_score'] for m in models.values()]) if models else 0
            },
            'output_files': {
                'report_path': report_path,
                'charts_directory': output_dir or Path(self.csv_path).parent / "analysis_charts"
            }
        }

        print("🎉 完整分析流程执行完成!")
        print(f"📄 报告路径: {summary['output_files']['report_path']}")
        print(f"📊 图表目录: {summary['output_files']['charts_directory']}")

        return summary


def main():
    """主函数 - 示例使用"""
    print("🔬 体验模式配置分析工具启动...")

    try:
        # 创建分析器实例
        analyzer = ExperienceConfigAnalyzer()

        # 运行完整分析
        results = analyzer.run_complete_analysis()

        print(f"\n✨ 分析完成总结:")
        print(f"   📝 处理数据: {results['data_summary']['total_records']:,}条记录")
        print(f"   🎯 唯一配置: {results['data_summary']['unique_configs']}种")
        print(f"   📊 目标指标: {results['data_summary']['target_metrics']}个")
        print(f"   🤖 最佳模型R²: {results['analysis_results']['best_model_r2']:.3f}")

    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()