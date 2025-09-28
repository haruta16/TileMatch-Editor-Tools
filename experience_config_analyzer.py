#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
体验模式配置[x,y,z]对TileMatch指标影响的深度分析工具
升级版：位置独立效应、交互效应、动态影响、机制分析

作者: Claude Code Assistant
升级时间: 2025-01-27
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
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# 设置中文字体和样式 - 修复中文显示问题
import sys
# 检查操作系统并设置合适的中文字体
if sys.platform.startswith('win'):
    # Windows系统字体配置
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
elif sys.platform.startswith('darwin'):
    # macOS系统字体配置
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
else:
    # Linux系统字体配置
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 10
plt.rcParams['figure.max_open_warning'] = 0  # 关闭图表数量警告
# 设置matplotlib后端为Agg，避免GUI相关问题
import matplotlib
matplotlib.use('Agg')
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class ExperienceConfigAnalyzer:
    """体验模式配置分析器 - 深度分析[x,y,z]位置影响机制"""

    def __init__(self, csv_path: str = None, csv_directory: str = None):
        """初始化分析器

        Args:
            csv_path: 单个CSV数据文件路径
            csv_directory: CSV文件目录路径，用于批量处理多个文件
        """
        self.csv_path = csv_path
        self.csv_directory = csv_directory or self._find_csv_directory()
        self.data = None
        self.features = None
        self.csv_files = []
        self.target_metrics = [
            'DifficultyScore', 'PeakDockCount', 'PressureValueMean',
            'PressureValueMax', 'PressureValueStdDev', 'FinalDifficulty',
            'InitialMinCost', 'DifficultyPosition'
        ]
        self.results = {}

    def _find_csv_directory(self) -> str:
        """自动查找CSV文件目录"""
        current_dir = Path(__file__).parent

        # 优先查找analysis_charts目录
        analysis_charts_dir = current_dir / "BattleAnalysisResults" / "analysis_charts"
        if analysis_charts_dir.exists():
            csv_files = list(analysis_charts_dir.glob("*.csv"))
            if csv_files:
                print(f"检测到analysis_charts目录，包含{len(csv_files)}个CSV文件")
                return str(analysis_charts_dir)

        # 查找BattleAnalysisResults根目录
        results_dir = current_dir / "BattleAnalysisResults"
        if results_dir.exists():
            csv_files = list(results_dir.glob("*.csv"))
            if csv_files:
                print(f"🔍 检测到BattleAnalysisResults目录，包含{len(csv_files)}个CSV文件")
                return str(results_dir)

        # 查找当前目录
        csv_files = list(current_dir.glob("*.csv"))
        if csv_files:
            print(f"🔍 检测到当前目录，包含{len(csv_files)}个CSV文件")
            return str(current_dir)

        raise FileNotFoundError("未找到包含CSV文件的目录，请确保数据文件存在")

    def load_and_preprocess(self) -> 'ExperienceConfigAnalyzer':
        """加载数据并进行预处理"""
        print("📊 加载和预处理数据...", flush=True)

        if self.csv_path:
            # 单文件模式
            self.data = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"✅ 单文件数据加载完成: {len(self.data)}条记录")
        else:
            # 多文件模式 - 批量加载
            self._load_multiple_csv_files()

        # 解析体验模式配置
        self._parse_experience_config()

        # 生成特征工程
        self._create_features()

        # 数据清洗
        self._clean_data()

        print(f"📈 预处理完成: {len(self.data)}条有效记录, {len(self.features.columns)}个特征")
        return self

    def _load_multiple_csv_files(self):
        """批量加载多个CSV文件，使用最简单的方案"""
        csv_dir = Path(self.csv_directory)
        self.csv_files = sorted(list(csv_dir.glob("*.csv")))

        if not self.csv_files:
            raise FileNotFoundError(f"在目录 {csv_dir} 中未找到CSV文件")

        print(f"🔍 发现{len(self.csv_files)}个CSV文件，准备批量加载...", flush=True)

        # 简单直接加载，一个文件一个文件处理
        data_list = []
        total_rows = 0

        for i, csv_file in enumerate(self.csv_files):
            print(f"   正在加载 {csv_file.name}... ({i+1}/{len(self.csv_files)})", flush=True)
            sys.stdout.flush()

            try:
                # 直接读取，不分块
                file_data = pd.read_csv(csv_file, encoding='utf-8')
                data_list.append(file_data)
                total_rows += len(file_data)
                print(f"     ✅ {csv_file.name}: {len(file_data)}行")

            except Exception as e:
                print(f"     ❌ 加载{csv_file.name}失败: {str(e)}")
                continue

        if not data_list:
            raise ValueError("所有CSV文件都加载失败")

        # 合并所有数据
        print("🔗 合并所有数据...", flush=True)
        self.data = pd.concat(data_list, ignore_index=True)

        print(f"✅ 批量数据加载完成: 总计{total_rows}条记录，合并后{len(self.data)}条记录")

    def _parse_experience_config(self):
        """解析体验模式配置[1,2,3]格式，使用最简单的实现"""
        # 检查ExperienceMode列是否存在
        if 'ExperienceMode' not in self.data.columns:
            print("⚠️ 未找到ExperienceMode列，使用默认配置")
            self.data['pos1'] = 1
            self.data['pos2'] = 2
            self.data['pos3'] = 3
            return

        print("🔧 解析体验模式配置...", flush=True)

        # 使用最简单的解析方式，避免复杂的apply操作
        pos1_list = []
        pos2_list = []
        pos3_list = []

        for config_str in self.data['ExperienceMode']:
            try:
                if pd.isna(config_str):
                    pos1_list.append(1)
                    pos2_list.append(2)
                    pos3_list.append(3)
                    continue

                # 简单字符串处理
                clean_str = str(config_str).strip('[]"()')
                numbers = [int(x.strip()) for x in clean_str.split(',')]

                if len(numbers) >= 3:
                    pos1_list.append(numbers[0])
                    pos2_list.append(numbers[1])
                    pos3_list.append(numbers[2])
                else:
                    pos1_list.append(1)
                    pos2_list.append(2)
                    pos3_list.append(3)

            except:
                pos1_list.append(1)
                pos2_list.append(2)
                pos3_list.append(3)

        # 直接赋值，不使用apply
        self.data['pos1'] = pos1_list
        self.data['pos2'] = pos2_list
        self.data['pos3'] = pos3_list

        # 验证解析结果
        unique_configs = len(self.data[['pos1', 'pos2', 'pos3']].drop_duplicates())
        print(f"   解析完成：发现{unique_configs}种不同配置")

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

        # 如果存在包含失败数据的数据集，也为其创建特征
        if hasattr(self, 'data_with_failures') and len(self.data_with_failures) > len(self.data):
            features_with_failures = self.data_with_failures[['pos1', 'pos2', 'pos3']].copy()
            # 添加基本特征
            features_with_failures['config_mean'] = (self.data_with_failures['pos1'] + self.data_with_failures['pos2'] + self.data_with_failures['pos3']) / 3
            self.features_with_failures = features_with_failures

    def _clean_data(self):
        """数据清洗，增强边界条件检查"""
        initial_count = len(self.data)
        print(f"🧹 开始数据清洗，初始记录数: {initial_count}")

        # 保存原始索引用于特征数据同步
        original_indices = self.data.index.copy()

        # 检查GameCompleted列是否存在
        if 'GameCompleted' in self.data.columns:
            # 对于非成功率分析，移除GameCompleted=False的记录
            # 保留原始数据用于后续成功率计算
            self.data_with_failures = self.data.copy()  # 保存包含失败游戏的完整数据

            completed_mask = (self.data['GameCompleted'] == True) | (self.data['GameCompleted'] == 'True') | (self.data['GameCompleted'] == 1)
            self.data = self.data[completed_mask].copy()
            print(f"   ✅ 完成游戏过滤: {len(self.data)}条记录用于主分析，{len(self.data_with_failures)}条记录保留用于成功率分析")
        else:
            print("   ⚠️ 未找到GameCompleted列，跳过完成状态过滤")
            self.data_with_failures = self.data.copy()  # 备份数据

        # 移除目标指标为空值的记录
        metrics_found = []
        for metric in self.target_metrics:
            if metric in self.data.columns:
                metrics_found.append(metric)
                before_count = len(self.data)
                self.data = self.data[self.data[metric].notna()].copy()
                if before_count != len(self.data):
                    print(f"   {metric}空值过滤: {len(self.data)}条记录")

        if not metrics_found:
            print("   ⚠️ 未找到任何目标指标列，数据可能存在问题")
        else:
            print(f"   找到{len(metrics_found)}个有效指标: {', '.join(metrics_found)}")

        # 移除配置解析异常的记录
        invalid_config_mask = (
            (self.data['pos1'] < 1) | (self.data['pos1'] > 9) |
            (self.data['pos2'] < 1) | (self.data['pos2'] > 9) |
            (self.data['pos3'] < 1) | (self.data['pos3'] > 9)
        )
        if invalid_config_mask.any():
            invalid_count = invalid_config_mask.sum()
            self.data = self.data[~invalid_config_mask].copy()
            print(f"   配置异常过滤: 移除{invalid_count}条，剩余{len(self.data)}条记录")

        # 安全的特征数据重新索引
        try:
            # 找到特征数据和清洗后数据的交集索引
            valid_indices = self.data.index.intersection(self.features.index)

            if len(valid_indices) == len(self.data):
                # 完美匹配，直接重新索引
                self.features = self.features.loc[valid_indices].copy()
            else:
                # 不完美匹配，重新生成特征数据以保证一致性
                print(f"   ⚠️ 特征数据索引不匹配，重新生成特征")
                self._create_features()

        except (KeyError, IndexError) as e:
            # 索引严重不匹配，重新生成
            print(f"   ❌ 特征数据索引错误，重新生成: {str(e)}")
            self._create_features()

        final_count = len(self.data)
        filtered_ratio = (initial_count - final_count) / initial_count * 100 if initial_count > 0 else 0
        print(f"🧹 数据清洗完成: {initial_count} -> {final_count}条记录 (过滤{filtered_ratio:.1f}%)")

        # 数据质量检查
        if final_count < initial_count * 0.1:  # 如果过滤掉超过90%的数据
            print(f"⚠️ 警告: 过滤比例过高({filtered_ratio:.1f}%)，请检查数据质量")

        if final_count < 100:  # 如果最终数据量过少
            print(f"⚠️ 警告: 有效数据量过少({final_count}条)，分析结果可能不可靠")

    def correlation_analysis(self) -> Dict:
        """基础相关性分析"""
        print("🔗 执行基础相关性分析...", flush=True)

        correlations = {}

        # 位置独立相关性
        position_corrs = {}
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_corrs = {}
            for metric in self.target_metrics:
                if metric in self.data.columns:
                    corr, p_value = pearsonr(self.features[pos], self.data[metric])
                    pos_corrs[metric] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': abs(corr)
                    }
            position_corrs[pos] = pos_corrs

        correlations['position_correlations'] = position_corrs

        # 特征重要性
        feature_importance = {}
        for metric in self.target_metrics:
            if metric in self.data.columns:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(self.features, self.data[metric])
                importance_dict = dict(zip(self.features.columns, rf.feature_importances_))
                feature_importance[metric] = dict(sorted(importance_dict.items(),
                                                       key=lambda x: x[1], reverse=True))

        correlations['feature_importance'] = feature_importance
        self.results['correlations'] = correlations
        print("✅ 基础相关性分析完成")
        return correlations

    def difficulty_position_analysis(self) -> Dict:
        """DifficultyPosition位置影响分析 - 分析不同体验配置在不同位置对DifficultyPosition的影响"""
        print("🏆 执行DifficultyPosition位置影响分析...", flush=True)

        position_effects = {}

        if 'DifficultyPosition' not in self.data.columns:
            print("   ⚠️ 无DifficultyPosition列，跳过此分析")
            return {}

        # 分析每个位置对DifficultyPosition的影响
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_effects = {}

            # 计算每个数值的DifficultyPosition平均值
            value_effects = {}
            for value in range(1, 10):
                subset = self.data[self.features[pos] == value]
                if len(subset) > 5:  # 确保样本量足够
                    difficulty_pos = subset['DifficultyPosition'].dropna()
                    if len(difficulty_pos) > 0:
                        value_effects[value] = {
                            'mean': difficulty_pos.mean(),
                            'std': difficulty_pos.std(),
                            'median': difficulty_pos.median(),
                            'count': len(difficulty_pos),
                            'min': difficulty_pos.min(),
                            'max': difficulty_pos.max()
                        }

            pos_effects['value_effects'] = value_effects

            # 计算相关性
            if len(self.features[pos]) > 0 and len(self.data['DifficultyPosition'].dropna()) > 0:
                corr, p_value = pearsonr(self.features[pos], self.data['DifficultyPosition'].fillna(0))
                pos_effects['correlation'] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

            # 找出最佳和最差配置（基于0-1范围：0=最前，0.99=最后，1=无难点）
            if value_effects:
                # 分离不同类型的配置
                no_difficulty_configs = {k: v for k, v in value_effects.items() if abs(v['mean'] - 1.0) < 0.01}  # 约等于1，无难点
                early_difficulty_configs = {k: v for k, v in value_effects.items() if 0 <= v['mean'] < 0.3}  # 前期难点
                mid_difficulty_configs = {k: v for k, v in value_effects.items() if 0.3 <= v['mean'] < 0.7}  # 中期难点
                late_difficulty_configs = {k: v for k, v in value_effects.items() if 0.7 <= v['mean'] < 1.0}  # 后期难点

                # 评估最佳配置的逻辑
                if mid_difficulty_configs:
                    # 中期难点是最理想的，选择中期难点中相对靠后的
                    best_value = max(mid_difficulty_configs.items(), key=lambda x: x[1]['mean'])
                    best_description = '中期难点（理想节奏）'
                elif late_difficulty_configs:
                    # 后期难点次之，选择适中的后期难点
                    best_value = min(late_difficulty_configs.items(), key=lambda x: x[1]['mean'])
                    best_description = '后期难点（渐进式挑战）'
                elif no_difficulty_configs:
                    # 无明显难点也可以接受
                    best_value = list(no_difficulty_configs.items())[0]
                    best_description = '无明显难点（平均体验）'
                elif early_difficulty_configs:
                    # 如果只有前期难点，选择相对靠后的
                    best_value = max(early_difficulty_configs.items(), key=lambda x: x[1]['mean'])
                    best_description = '前期难点（相对较晚）'
                else:
                    # 兜底
                    best_value = max(value_effects.items(), key=lambda x: x[1]['mean'])
                    best_description = '相对最佳'

                # 评估最差配置的逻辑
                if early_difficulty_configs:
                    # 前期难点是最差的，选择最早的
                    worst_value = min(early_difficulty_configs.items(), key=lambda x: x[1]['mean'])
                    worst_description = '前期难点（开局挫败）'
                elif value_effects:
                    # 如果没有前期难点，选择最不理想的
                    worst_value = min(value_effects.items(), key=lambda x: x[1]['mean'])
                    worst_description = '相对最差'
                else:
                    worst_value = best_value
                    worst_description = '相对最差'

                pos_effects['best_config'] = {
                    'value': best_value[0],
                    'mean_difficulty_position': best_value[1]['mean'],
                    'sample_count': best_value[1]['count'],
                    'description': best_description
                }

                pos_effects['worst_config'] = {
                    'value': worst_value[0],
                    'mean_difficulty_position': worst_value[1]['mean'],
                    'sample_count': worst_value[1]['count'],
                    'description': worst_description
                }

                # 添加各类配置的统计
                config_categories = {
                    'no_difficulty': (no_difficulty_configs, '无明显难点'),
                    'early_difficulty': (early_difficulty_configs, '前期难点'),
                    'mid_difficulty': (mid_difficulty_configs, '中期难点'),
                    'late_difficulty': (late_difficulty_configs, '后期难点')
                }

                for category_name, (configs, description) in config_categories.items():
                    if configs:
                        pos_effects[f'{category_name}_configs'] = {
                            'count': len(configs),
                            'values': list(configs.keys()),
                            'description': description,
                            'avg_position': sum(v['mean'] for v in configs.values()) / len(configs)
                        }

            position_effects[pos] = pos_effects

        # 交互效应分析
        interaction_effects = {}
        for pos_pair in [('pos1', 'pos2'), ('pos1', 'pos3'), ('pos2', 'pos3')]:
            pair_key = f"{pos_pair[0]}×{pos_pair[1]}"

            # 计算交互项对DifficultyPosition的影响
            interaction_feature = self.features[pos_pair[0]] * self.features[pos_pair[1]]

            # 基础模型 vs 交互模型
            X_base = self.features[[pos_pair[0], pos_pair[1]]]
            X_inter = X_base.copy()
            X_inter['interaction'] = interaction_feature

            y_valid = self.data['DifficultyPosition'].dropna()
            X_base_valid = X_base.loc[y_valid.index]
            X_inter_valid = X_inter.loc[y_valid.index]

            if len(y_valid) > 10:
                try:
                    rf_base = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf_base.fit(X_base_valid, y_valid)
                    r2_base = rf_base.score(X_base_valid, y_valid)

                    rf_inter = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf_inter.fit(X_inter_valid, y_valid)
                    r2_inter = rf_inter.score(X_inter_valid, y_valid)

                    interaction_effects[pair_key] = {
                        'r2_base': r2_base,
                        'r2_interaction': r2_inter,
                        'interaction_gain': r2_inter - r2_base,
                        'sample_count': len(y_valid)
                    }
                except Exception as e:
                    print(f"   ⚠️ {pair_key}交互分析失败: {str(e)}")

        position_effects['interaction_effects'] = interaction_effects

        self.results['difficulty_position_effects'] = position_effects
        print("✅ DifficultyPosition位置影响分析完成")
        return position_effects

    def position_independent_analysis(self) -> Dict:
        """位置独立影响分析 - 控制其他变量分析单个位置的纯净效应"""
        print("🎯 执行位置独立影响分析...", flush=True)

        independent_effects = {}

        for pos in ['pos1', 'pos2', 'pos3']:
            pos_effects = {}

            for metric in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']:
                if metric in self.data.columns:
                    # 计算边际效应：pos每变化1单位对metric的影响
                    marginal_effects = []

                    # 分析不同数值区间的效应
                    for value in range(1, 10):
                        subset = self.data[self.features[pos] == value]
                        if len(subset) > 5:  # 确保样本量足够
                            avg_metric = subset[metric].mean()
                            marginal_effects.append((value, avg_metric))

                    if len(marginal_effects) > 2:
                        # 计算边际递增效应
                        marginal_diffs = []
                        for i in range(1, len(marginal_effects)):
                            diff = marginal_effects[i][1] - marginal_effects[i-1][1]
                            marginal_diffs.append(diff)

                        pos_effects[metric] = {
                            'marginal_effects': marginal_effects,
                            'avg_marginal_diff': np.mean(marginal_diffs),
                            'marginal_volatility': np.std(marginal_diffs)
                        }

            independent_effects[pos] = pos_effects

        self.results['independent_effects'] = independent_effects
        print("✅ 位置独立影响分析完成")
        return independent_effects

    def interaction_analysis(self) -> Dict:
        """位置交互效应分析 - 分析位置间相互作用"""
        print("🔄 执行位置交互效应分析...", flush=True)

        interaction_effects = {}

        # 双位置交互分析
        for pos_pair in [('pos1', 'pos2'), ('pos1', 'pos3'), ('pos2', 'pos3')]:
            pair_key = f"{pos_pair[0]}×{pos_pair[1]}"
            pair_effects = {}

            for metric in ['DifficultyScore', 'PeakDockCount']:
                if metric in self.data.columns:
                    # 创建交互特征
                    interaction_feature = self.features[pos_pair[0]] * self.features[pos_pair[1]]

                    # 比较有交互项vs无交互项的模型性能
                    X_base = self.features[[pos_pair[0], pos_pair[1]]]
                    X_inter = X_base.copy()
                    X_inter['interaction'] = interaction_feature

                    # 基础模型
                    rf_base = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf_base.fit(X_base, self.data[metric])
                    r2_base = rf_base.score(X_base, self.data[metric])

                    # 交互模型
                    rf_inter = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf_inter.fit(X_inter, self.data[metric])
                    r2_inter = rf_inter.score(X_inter, self.data[metric])

                    pair_effects[metric] = {
                        'r2_base': r2_base,
                        'r2_interaction': r2_inter,
                        'interaction_gain': r2_inter - r2_base,
                        'interaction_strength': abs(r2_inter - r2_base)
                    }

            interaction_effects[pair_key] = pair_effects

        self.results['interaction_effects'] = interaction_effects
        print("✅ 位置交互效应分析完成")
        return interaction_effects

    def dynamic_impact_analysis(self) -> Dict:
        """动态影响分析 - 分析位置在不同游戏阶段的差异化影响"""
        print("⏱️ 执行动态影响分析...", flush=True)

        dynamic_effects = {}

        # 解析DockAfterTrioMatch序列数据
        dock_sequences = []
        for dock_str in self.data['DockAfterTrioMatch'].fillna(''):
            if dock_str:
                try:
                    dock_values = [int(x) for x in str(dock_str).split(',')]
                    dock_sequences.append(dock_values)
                except:
                    dock_sequences.append([])
            else:
                dock_sequences.append([])

        # 分析各位置对不同游戏阶段的影响
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_dynamic = {}

            # 分析对游戏前期、中期、后期的差异化影响
            stage_effects = {'early': [], 'middle': [], 'late': []}

            for i, dock_seq in enumerate(dock_sequences):
                if len(dock_seq) >= 6:  # 确保序列足够长
                    seq_len = len(dock_seq)
                    early_avg = np.mean(dock_seq[:seq_len//3])
                    middle_avg = np.mean(dock_seq[seq_len//3:2*seq_len//3])
                    late_avg = np.mean(dock_seq[2*seq_len//3:])

                    pos_value = self.features.iloc[i][pos]
                    stage_effects['early'].append((pos_value, early_avg))
                    stage_effects['middle'].append((pos_value, middle_avg))
                    stage_effects['late'].append((pos_value, late_avg))

            # 计算各阶段的相关性
            for stage, values in stage_effects.items():
                if len(values) > 10:
                    pos_vals, dock_vals = zip(*values)
                    corr, _ = pearsonr(pos_vals, dock_vals)
                    pos_dynamic[f'{stage}_correlation'] = corr

            dynamic_effects[pos] = pos_dynamic

        self.results['dynamic_effects'] = dynamic_effects
        print("✅ 动态影响分析完成")
        return dynamic_effects

    def mechanism_analysis(self) -> Dict:
        """影响机制分析 - 分析位置影响指标的中介路径"""
        print("🔍 执行影响机制分析...", flush=True)

        mechanism_effects = {}

        # 对于每个位置，分析其对DifficultyScore的中介路径
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_mechanism = {}

            # 直接效应：pos -> DifficultyScore
            if 'DifficultyScore' in self.data.columns:
                direct_corr, _ = pearsonr(self.features[pos], self.data['DifficultyScore'])
                pos_mechanism['direct_effect'] = direct_corr

                # 中介效应分析：pos -> 中介变量 -> DifficultyScore
                mediators = ['PressureValueMean', 'PeakDockCount', 'PressureValueMax']
                mediation_effects = {}

                for mediator in mediators:
                    if mediator in self.data.columns:
                        # pos -> mediator
                        corr_pm, _ = pearsonr(self.features[pos], self.data[mediator])
                        # mediator -> DifficultyScore
                        corr_md, _ = pearsonr(self.data[mediator], self.data['DifficultyScore'])
                        # 中介效应强度 = 两个相关系数的乘积
                        mediation_strength = corr_pm * corr_md

                        mediation_effects[mediator] = {
                            'pos_to_mediator': corr_pm,
                            'mediator_to_target': corr_md,
                            'mediation_strength': mediation_strength
                        }

                pos_mechanism['mediation_effects'] = mediation_effects

                # 找出最强的中介路径
                if mediation_effects:
                    strongest_mediator = max(mediation_effects.items(),
                                           key=lambda x: abs(x[1]['mediation_strength']))
                    pos_mechanism['strongest_mediation_path'] = {
                        'mediator': strongest_mediator[0],
                        'strength': strongest_mediator[1]['mediation_strength']
                    }

            mechanism_effects[pos] = pos_mechanism

        self.results['mechanism_effects'] = mechanism_effects
        print("✅ 影响机制分析完成")
        return mechanism_effects

    def value_specific_analysis(self, target_value: int = None) -> Dict:
        """单一数值深度分析 - 分析特定数值(如x=5)在不同位置的完整影响"""
        print(f"🎯 执行单一数值深度分析 (目标数值: {target_value or '全部'})...")

        value_effects = {}
        values_to_analyze = [target_value] if target_value else range(1, 10)

        for value in values_to_analyze:
            value_key = f"value_{value}"
            value_data = {}

            for pos in ['pos1', 'pos2', 'pos3']:
                pos_data = {}

                # 筛选该数值在该位置的数据
                mask = self.features[pos] == value
                subset = self.data[mask]

                if len(subset) < 10:  # 样本量太少跳过
                    continue

                # DifficultyScore影响分析
                if 'DifficultyScore' in subset.columns:
                    difficulty_stats = {
                        'mean': subset['DifficultyScore'].mean(),
                        'std': subset['DifficultyScore'].std(),
                        'median': subset['DifficultyScore'].median(),
                        'min': subset['DifficultyScore'].min(),
                        'max': subset['DifficultyScore'].max(),
                        'count': len(subset)
                    }
                    pos_data['difficulty_impact'] = difficulty_stats

                # 胜率分析 - 使用包含失败游戏的完整数据
                if 'GameCompleted' in self.data_with_failures.columns:
                    # 从完整数据中筛选当前配置的子集
                    # 安全的索引方式处理包含失败数据的完整数据集
                    if hasattr(self, 'data_with_failures'):
                        if hasattr(self, 'features_with_failures'):
                            # 使用包含失败数据的特征
                            full_subset = self.data_with_failures[self.features_with_failures[pos] == value]
                        else:
                            # 直接从原始数据中解析配置进行筛选
                            def get_pos_value(exp_mode_str):
                                try:
                                    clean_str = str(exp_mode_str).strip('[]"()')
                                    numbers = [int(x.strip()) for x in clean_str.split(',')]
                                    if len(numbers) >= 3:
                                        pos_map = {'pos1': numbers[0], 'pos2': numbers[1], 'pos3': numbers[2]}
                                        return pos_map.get(pos, 0)
                                    return 0
                                except:
                                    return 0

                            exp_mode_filter = self.data_with_failures['ExperienceMode'].astype(str).apply(
                                lambda x: get_pos_value(x) == value
                            )
                            full_subset = self.data_with_failures[exp_mode_filter]
                    else:
                        full_subset = subset
                    if len(full_subset) > 0:
                        # 正确处理GameCompleted列的不同格式
                        completed_mask = (full_subset['GameCompleted'] == True) | (full_subset['GameCompleted'] == 'True') | (full_subset['GameCompleted'] == 1)
                        win_rate = completed_mask.mean()
                        pos_data['win_rate'] = {
                            'success_rate': win_rate,
                            'failure_rate': 1 - win_rate,
                            'total_games': len(full_subset)
                        }

                # DockAfterTrioMatch序列分析
                if 'DockAfterTrioMatch' in subset.columns:
                    dock_analysis = self._analyze_dock_sequences(subset['DockAfterTrioMatch'])
                    pos_data['dock_impact'] = dock_analysis

                # PressureValues分析
                pressure_analysis = {}
                for pressure_col in ['PressureValueMean', 'PressureValueMax', 'PressureValueStdDev']:
                    if pressure_col in subset.columns:
                        pressure_analysis[pressure_col] = {
                            'mean': subset[pressure_col].mean(),
                            'std': subset[pressure_col].std(),
                            'median': subset[pressure_col].median()
                        }
                pos_data['pressure_impact'] = pressure_analysis

                value_data[pos] = pos_data

            value_effects[value_key] = value_data

        self.results['value_specific_effects'] = value_effects
        print("✅ 单一数值深度分析完成")
        return value_effects

    def _analyze_dock_sequences(self, dock_series) -> Dict:
        """分析DockAfterTrioMatch序列的辅助方法"""
        sequences = []
        for dock_str in dock_series.fillna(''):
            if dock_str:
                try:
                    dock_values = [int(x) for x in str(dock_str).split(',')]
                    sequences.append(dock_values)
                except:
                    continue

        if not sequences:
            return {}

        # 计算序列统计
        seq_lengths = [len(seq) for seq in sequences]

        # 分析不同阶段的平均Dock值
        early_vals, middle_vals, late_vals = [], [], []
        for seq in sequences:
            if len(seq) >= 6:
                third = len(seq) // 3
                early_vals.extend(seq[:third])
                middle_vals.extend(seq[third:2*third])
                late_vals.extend(seq[2*third:])

        return {
            'total_sequences': len(sequences),
            'phase_analysis': {
                'early_phase': {'mean': np.mean(early_vals) if early_vals else 0, 'count': len(early_vals)},
                'middle_phase': {'mean': np.mean(middle_vals) if middle_vals else 0, 'count': len(middle_vals)},
                'late_phase': {'mean': np.mean(late_vals) if late_vals else 0, 'count': len(late_vals)}
            }
        }

    def gradient_effect_analysis(self) -> Dict:
        """数值梯度效应分析 - 分析1-9数值的连续影响变化"""
        print("📈 执行数值梯度效应分析...")

        gradient_effects = {}

        for pos in ['pos1', 'pos2', 'pos3']:
            pos_gradients = {}

            # 主要指标的梯度分析
            for metric in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']:
                if metric in self.data.columns:

                    # 计算每个数值的平均指标值
                    value_means = []
                    value_counts = []

                    for value in range(1, 10):
                        subset = self.data[self.features[pos] == value]
                        if len(subset) > 0:
                            value_means.append(subset[metric].mean())
                            value_counts.append(len(subset))
                        else:
                            value_means.append(np.nan)
                            value_counts.append(0)

                    # 计算梯度(相邻数值间的差值)
                    gradients = []
                    for i in range(1, len(value_means)):
                        if not (np.isnan(value_means[i]) or np.isnan(value_means[i-1])):
                            gradient = value_means[i] - value_means[i-1]
                            gradients.append(gradient)
                        else:
                            gradients.append(np.nan)

                    # 识别最大梯度变化点(临界点)
                    abs_gradients = [abs(g) for g in gradients if not np.isnan(g)]
                    if abs_gradients:
                        max_gradient_idx = gradients.index(max(gradients, key=abs))
                        critical_point = max_gradient_idx + 2  # +2因为梯度是差值,对应后一个数值
                    else:
                        critical_point = None

                    pos_gradients[metric] = {
                        'value_means': value_means,
                        'gradients': gradients,
                        'critical_point': critical_point,
                        'max_gradient': max(abs_gradients) if abs_gradients else 0,
                        'avg_gradient': np.mean([g for g in gradients if not np.isnan(g)]) if gradients else 0
                    }

            gradient_effects[pos] = pos_gradients

        self.results['gradient_effects'] = gradient_effects
        print("✅ 数值梯度效应分析完成")
        return gradient_effects

    def dock_sequence_deep_analysis(self) -> Dict:
        """DockAfterTrioMatch序列深度分析"""
        print("🚢 执行Dock序列深度分析...")

        dock_deep_analysis = {}

        # 解析所有序列数据 - 使用包含失败游戏的完整数据进行成功率计算
        data_for_analysis = self.data_with_failures if hasattr(self, 'data_with_failures') else self.data
        all_sequences = []
        sequence_metadata = []

        for idx, row in data_for_analysis.iterrows():
            dock_str = row.get('DockAfterTrioMatch', '')
            if dock_str and str(dock_str) != 'nan':
                try:
                    dock_values = [int(x) for x in str(dock_str).split(',')]
                    all_sequences.append(dock_values)

                    # 需要从特征数据中获取pos配置（如果索引对应的话）
                    if idx in self.features.index:
                        pos_config = {
                            'pos1': self.features.loc[idx, 'pos1'],
                            'pos2': self.features.loc[idx, 'pos2'],
                            'pos3': self.features.loc[idx, 'pos3']
                        }
                    else:
                        # 重新解析配置（备用方案）
                        exp_mode = row.get('ExperienceMode', '[1,2,3]')
                        try:
                            clean_str = str(exp_mode).strip('[]"()')
                            numbers = [int(x.strip()) for x in clean_str.split(',')]
                            if len(numbers) >= 3:
                                pos_config = {'pos1': numbers[0], 'pos2': numbers[1], 'pos3': numbers[2]}
                            else:
                                pos_config = {'pos1': 1, 'pos2': 2, 'pos3': 3}
                        except:
                            pos_config = {'pos1': 1, 'pos2': 2, 'pos3': 3}

                    sequence_metadata.append({
                        **pos_config,
                        'difficulty': row.get('DifficultyScore', 0),
                        'completed': row.get('GameCompleted', False)
                    })
                except:
                    continue

        if not all_sequences:
            return {}

        # 按位置分组分析
        for pos in ['pos1', 'pos2', 'pos3']:
            pos_analysis = {}

            for value in range(1, 10):
                value_sequences = []
                value_metadata = []

                for i, meta in enumerate(sequence_metadata):
                    if meta[pos] == value:
                        value_sequences.append(all_sequences[i])
                        value_metadata.append(meta)

                if len(value_sequences) < 5:  # 样本量太少
                    continue

                # 序列长度分析
                lengths = [len(seq) for seq in value_sequences]

                # 序列模式分析
                patterns = self._identify_dock_patterns(value_sequences)

                # 危险时刻分析(Dock>=6的时刻)
                danger_analysis = self._analyze_danger_moments(value_sequences)

                # 成功率与序列特征关系
                completed_values = []
                for meta in value_metadata:
                    completed = meta['completed']
                    # 处理不同的GameCompleted格式
                    if completed == True or completed == 'True' or completed == 1:
                        completed_values.append(1)
                    else:
                        completed_values.append(0)

                success_rate = np.mean(completed_values) if completed_values else 0

                pos_analysis[f'value_{value}'] = {
                    'sequence_count': len(value_sequences),
                    'success_rate': success_rate,
                    'patterns': patterns,
                    'danger_analysis': danger_analysis
                }

            dock_deep_analysis[pos] = pos_analysis

        self.results['dock_deep_analysis'] = dock_deep_analysis
        print("✅ Dock序列深度分析完成")
        return dock_deep_analysis

    def _identify_dock_patterns(self, sequences) -> Dict:
        """识别Dock序列模式"""
        if not sequences:
            return {}

        # 计算平均序列
        max_length = max(len(seq) for seq in sequences)
        avg_sequence = []

        for i in range(max_length):
            values_at_i = [seq[i] for seq in sequences if len(seq) > i]
            if values_at_i:
                avg_sequence.append(np.mean(values_at_i))

        # 识别模式类型
        if len(avg_sequence) < 3:
            return {'pattern_type': 'insufficient_data'}

        # 判断趋势
        early_avg = np.mean(avg_sequence[:len(avg_sequence)//3])
        late_avg = np.mean(avg_sequence[2*len(avg_sequence)//3:])

        if late_avg > early_avg + 0.5:
            pattern_type = 'increasing_pressure'
        elif late_avg < early_avg - 0.5:
            pattern_type = 'decreasing_pressure'
        else:
            pattern_type = 'stable_pressure'

        return {
            'pattern_type': pattern_type,
            'avg_sequence': avg_sequence[:10],  # 只保存前10个点
            'early_avg': early_avg,
            'late_avg': late_avg
        }

    def _analyze_danger_moments(self, sequences) -> Dict:
        """分析危险时刻(Dock>=6)"""
        if not sequences:
            return {}

        danger_counts = []
        max_danger_levels = []

        for seq in sequences:
            danger_count = sum(1 for val in seq if val >= 6)
            max_danger = max(seq) if seq else 0

            danger_counts.append(danger_count)
            max_danger_levels.append(max_danger)

        return {
            'avg_danger_moments': np.mean(danger_counts),
            'max_danger_level': np.mean(max_danger_levels),
            'danger_frequency': np.mean([d > 0 for d in danger_counts])
        }

    def pressure_dynamics_analysis(self) -> Dict:
        """压力动态分析"""
        print("⚡ 执行压力动态分析...")

        pressure_dynamics = {}

        pressure_columns = ['PressureValueMean', 'PressureValueMax', 'PressureValueStdDev']

        for pos in ['pos1', 'pos2', 'pos3']:
            pos_dynamics = {}

            for value in range(1, 10):
                subset = self.data[self.features[pos] == value]

                if len(subset) < 10:
                    continue

                value_dynamics = {}

                for pressure_col in pressure_columns:
                    if pressure_col in subset.columns:
                        pressure_values = subset[pressure_col].dropna()

                        if len(pressure_values) > 0:
                            # 基础统计
                            stats = {
                                'mean': pressure_values.mean(),
                                'std': pressure_values.std(),
                                'median': pressure_values.median(),
                                'q75': pressure_values.quantile(0.75),
                                'q95': pressure_values.quantile(0.95),
                                'max': pressure_values.max()
                            }

                            # 压力分布分析
                            low_pressure = (pressure_values < pressure_values.quantile(0.33)).sum()
                            high_pressure = (pressure_values > pressure_values.quantile(0.67)).sum()

                            pressure_distribution = {
                                'low_pressure_count': low_pressure,
                                'high_pressure_count': high_pressure,
                                'extreme_pressure_count': (pressure_values > pressure_values.quantile(0.9)).sum()
                            }

                            value_dynamics[pressure_col] = {
                                'statistics': stats,
                                'distribution': pressure_distribution
                            }

                pos_dynamics[f'value_{value}'] = value_dynamics

            pressure_dynamics[pos] = pos_dynamics

        self.results['pressure_dynamics'] = pressure_dynamics
        print("✅ 压力动态分析完成")
        return pressure_dynamics

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

            # 训练测试分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 随机森林模型
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train_scaled, y_train)

            # 模型评估
            y_pred = rf_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # 特征重要性
            feature_importance = dict(zip(X.columns, rf_model.feature_importances_))

            models[metric] = {
                'model': rf_model,
                'scaler': scaler,
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance
            }

        self.results['models'] = models
        print(f"✅ 预测模型构建完成，共{len(models)}个模型")
        return models

    def run_complete_analysis(self, output_dir: str = None) -> Dict:
        """运行升级版完整分析流程"""
        print("🚀 开始深度体验配置影响分析...")

        # 数据加载与预处理
        self.load_and_preprocess()

        # 基础分析
        correlations = self.correlation_analysis()
        patterns = self.pattern_analysis()

        # 重要：新增DifficultyPosition影响分析
        difficulty_position_effects = self.difficulty_position_analysis()

        # 深度分析（新增）
        independent_effects = self.position_independent_analysis()
        interaction_effects = self.interaction_analysis()
        dynamic_effects = self.dynamic_impact_analysis()
        mechanism_effects = self.mechanism_analysis()

        # 新增深度分析方法
        value_specific_effects = self.value_specific_analysis()
        gradient_effects = self.gradient_effect_analysis()
        dock_deep_effects = self.dock_sequence_deep_analysis()
        pressure_dynamics = self.pressure_dynamics_analysis()

        # 机器学习建模
        models = self.build_prediction_models()

        # 创建增强可视化
        self.create_enhanced_visualizations(output_dir)

        # 生成深度报告
        report_path = self.generate_enhanced_report()

        # 输出关键发现
        self._print_key_findings()

        # 汇总结果
        summary = {
            'data_summary': {
                'total_records': len(self.data),
                'unique_configs': len(self.data[['pos1', 'pos2', 'pos3']].drop_duplicates()),
                'target_metrics': len(self.target_metrics)
            },
            'analysis_results': {
                'basic_analysis': len(correlations) > 0 and len(patterns) > 0,
                'advanced_analysis': len(independent_effects) > 0 and len(interaction_effects) > 0,
                'dynamic_analysis': len(dynamic_effects) > 0,
                'mechanism_analysis': len(mechanism_effects) > 0,
                'value_specific_analysis': len(value_specific_effects) > 0,
                'gradient_analysis': len(gradient_effects) > 0,
                'dock_deep_analysis': len(dock_deep_effects) > 0,
                'pressure_dynamics': len(pressure_dynamics) > 0,
                'models_completed': len(models) > 0,
                'best_model_r2': max([m['r2_score'] for m in models.values()]) if models else 0
            },
            'output_files': {
                'report_path': report_path,
                'charts_directory': output_dir or (
                    Path(self.csv_path).parent / "analysis_charts" if self.csv_path
                    else Path(self.csv_directory) / "analysis_output"
                )
            }
        }

        print("🎉 深度分析流程执行完成!")
        print(f"📄 报告路径: {summary['output_files']['report_path']}")
        print(f"📊 图表目录: {summary['output_files']['charts_directory']}")

        return summary

    def create_enhanced_visualizations(self, output_dir: str = None):
        """创建增强可视化图表"""
        print("📊 创建增强可视化图表...", flush=True)

        if output_dir is None:
            # 多文件模式使用csv_directory，单文件模式使用csv_path
            if self.csv_path:
                output_dir = Path(self.csv_path).parent / "analysis_charts"
            else:
                output_dir = Path(self.csv_directory) / "analysis_output"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 安全图表绘制列表
        charts = [
            ("位置相关性热力图", self._plot_position_correlation_heatmap),
            ("配置模式箱线图", self._plot_pattern_boxplots),
            ("特征重要性图", self._plot_feature_importance),
            ("配置分布图", self._plot_config_distribution),
            ("模型性能图", self._plot_model_performance),
            ("DifficultyPosition影响分析图", self._plot_difficulty_position_analysis),
            ("位置独立效应图", self._plot_independent_effects),
            ("交互效应图", self._plot_interaction_effects),
            ("机制分析图", self._plot_mechanism_analysis),
            ("数值影响矩阵热力图", self._plot_value_impact_heatmaps),
            ("数值梯度效应曲线", self._plot_gradient_curves),
            ("Dock序列模式图", self._plot_dock_sequence_patterns),
            ("压力动态分布图", self._plot_pressure_dynamics)
        ]

        successful_charts = 0
        for chart_name, chart_func in charts:
            try:
                print(f"   绘制 {chart_name}...", end='', flush=True)
                chart_func(output_path)
                successful_charts += 1
                print(f" ✅ 完成", flush=True)
            except Exception as e:
                print(f"   ❌ {chart_name} 失败: {str(e)}")
                continue

        print(f"✅ 增强可视化图表已保存到: {output_path}")
        print(f"📊 成功绘制: {successful_charts}/{len(charts)} 个图表")

    def _plot_independent_effects(self, output_path: Path):
        """绘制位置独立效应图"""
        if 'independent_effects' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, pos in enumerate(['pos1', 'pos2', 'pos3']):
            if pos in self.results['independent_effects']:
                effects = self.results['independent_effects'][pos]

                if 'DifficultyScore' in effects:
                    marginal_data = effects['DifficultyScore']['marginal_effects']
                    if marginal_data:
                        x_vals, y_vals = zip(*marginal_data)
                        axes[i].plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8)
                        axes[i].set_title(f'{pos} Marginal Effect on DifficultyScore')
                        axes[i].set_xlabel(f'{pos} Value')
                        axes[i].set_ylabel('DifficultyScore Mean')
                        axes[i].grid(True, alpha=0.3)

        plt.suptitle('Position Independent Effects Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'independent_effects.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_interaction_effects(self, output_path: Path):
        """绘制交互效应强度图，增加数值范围检查"""
        if 'interaction_effects' not in self.results:
            return

        pairs = list(self.results['interaction_effects'].keys())
        metrics = ['DifficultyScore', 'PeakDockCount']

        if not pairs:
            print("   ⚠️ 无交互效应数据，跳过绘制")
            return

        # 收集所有数值并检查范围
        all_strengths = []
        for pair in pairs:
            for metric in metrics:
                if metric in self.results['interaction_effects'][pair]:
                    strength = self.results['interaction_effects'][pair][metric]['interaction_strength']
                    if np.isfinite(strength):  # 检查数值有效性
                        all_strengths.append(abs(strength))

        if not all_strengths:
            print("   ⚠️ 无有效交互效应强度数据，跳过绘制")
            return

        # 检查数值范围合理性
        max_strength = max(all_strengths)
        if max_strength > 1e6:  # 异常大数值
            print(f"   ⚠️ 检测到异常大数值({max_strength:.2e})，跳过交互效应图绘制")
            return

        try:
            fig, ax = plt.subplots(figsize=(min(12, len(pairs) * 2), 6))

            x_pos = np.arange(len(pairs))
            width = 0.35

            for i, metric in enumerate(metrics):
                strengths = []
                for pair in pairs:
                    if metric in self.results['interaction_effects'][pair]:
                        strength = self.results['interaction_effects'][pair][metric]['interaction_strength']
                        # 数值安全检查
                        if np.isfinite(strength):
                            strengths.append(max(0, min(1, abs(strength))))  # 限制在[0,1]范围
                        else:
                            strengths.append(0)
                    else:
                        strengths.append(0)

                if any(s > 0 for s in strengths):  # 只绘制有数据的指标
                    bars = ax.bar(x_pos + i * width, strengths, width, label=metric, alpha=0.7)

                    # 添加数值标签
                    for bar, strength in zip(bars, strengths):
                        if strength > 0.001:  # 只显示显著值
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_strength*0.01,
                                   f'{strength:.3f}', ha='center', va='bottom', fontsize=9)

            ax.set_xlabel('Position Combination')
            ax.set_ylabel('Interaction Effect Strength')
            ax.set_title('Position Interaction Effects Analysis')
            ax.set_xticks(x_pos + width / 2)
            ax.set_xticklabels(pairs, rotation=45 if len(pairs) > 3 else 0)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 设置合理的Y轴范围
            ax.set_ylim(0, max(all_strengths) * 1.1)

            plt.tight_layout()
            plt.savefig(output_path / 'interaction_effects.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"   ❌ 交互效应图绘制内部错误: {str(e)}")
            plt.close('all')  # 确保清理资源

    def _plot_mechanism_analysis(self, output_path: Path):
        """绘制机制分析图"""
        if 'mechanism_effects' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, pos in enumerate(['pos1', 'pos2', 'pos3']):
            if pos in self.results['mechanism_effects']:
                mechanism = self.results['mechanism_effects'][pos]

                if 'mediation_effects' in mechanism:
                    mediators = list(mechanism['mediation_effects'].keys())
                    strengths = [mechanism['mediation_effects'][m]['mediation_strength']
                               for m in mediators]

                    colors = ['red' if s > 0 else 'blue' for s in strengths]
                    bars = axes[i].barh(mediators, [abs(s) for s in strengths], color=colors, alpha=0.7)

                    axes[i].set_title(f'{pos} Mediation Effect Analysis')
                    axes[i].set_xlabel('Mediation Effect Strength')

                    # 添加数值标签
                    for bar, strength in zip(bars, strengths):
                        axes[i].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                   f'{strength:.3f}', va='center', fontsize=9)

        plt.suptitle('影响机制中介路径分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'mechanism_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_enhanced_report(self, output_path: str = None) -> str:
        """生成增强版分析报告"""
        print("📋 生成深度分析报告...", flush=True)

        if output_path is None:
            # 多文件模式使用csv_directory，单文件模式使用csv_path
            if self.csv_path:
                output_path = Path(self.csv_path).parent / "enhanced_analysis_report.md"
            else:
                output_path = Path(self.csv_directory) / "enhanced_analysis_report.md"

        report = []
        report.append("# 体验模式配置[x,y,z]深度影响分析报告\n")
        report.append("## 📖 报告说明\n")
        report.append("本报告分析体验配置[x,y,z]三个位置的数值对游戏各项指标的影响。")
        report.append("体验配置是游戏难度调节的重要参数，通过分析不同配置下的游戏表现，")
        report.append("可以帮助优化游戏平衡性和玩家体验。\n")

        report.append("### 📊 分析维度说明")
        report.append("- **DifficultyScore**: 游戏难度评分，数值越高表示难度越大")
        report.append("- **PeakDockCount**: Dock区域瓦片数量峰值，反映游戏中的压力情况")
        report.append("- **PressureValueMean**: 平均压力值，衡量游戏整体压力水平")
        report.append("- **DifficultyPosition**: 关卡流程内难点出现的位置。数值范围0-1，其中0表示难点在最前面，0.99表示难点在最后面，1表示无明显难点位置")
        report.append("- **成功率**: 玩家完成游戏的比例，是最重要的体验指标\n")

        report.append("### 🎯 配置位置说明")
        report.append("- **pos1**: 配置数组第一个位置，主要影响游戏前期难度")
        report.append("- **pos2**: 配置数组第二个位置，主要影响游戏中期难度")
        report.append("- **pos3**: 配置数组第三个位置，主要影响游戏后期难度")
        report.append("- **数值范围**: 1-9，数值越小难度越低，数值越大难度越高\n")

        # 数据源信息
        if self.csv_path:
            report.append(f"**数据源**: {self.csv_path}")
        else:
            report.append(f"**数据源**: {self.csv_directory} ({len(self.csv_files)}个CSV文件)")

        report.append(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**样本数量**: {len(self.data):,}条\n")

        # 数据概览
        self._add_data_overview(report)

        # DifficultyPosition影响分析 (新增)
        self._add_difficulty_position_effects_report(report)

        # 位置独立效应分析
        self._add_independent_effects_report(report)

        # 交互效应分析
        self._add_interaction_effects_report(report)

        # 动态影响分析
        self._add_dynamic_effects_report(report)

        # 机制分析
        self._add_mechanism_effects_report(report)

        # 新增深度分析报告
        self._add_value_specific_report(report)
        self._add_gradient_effects_report(report)
        self._add_dock_sequence_report(report)
        self._add_pressure_dynamics_report(report)

        # 关键发现与建议
        self._add_key_findings_and_recommendations(report)

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ 深度分析报告已保存到: {output_path}")
        return str(output_path)
    

    def _add_data_overview(self, report):
        report.append("## 📊 数据概览\n")
        report.append(f"- 配置范围: pos1=[{self.features['pos1'].min()}-{self.features['pos1'].max()}], " +
                     f"pos2=[{self.features['pos2'].min()}-{self.features['pos2'].max()}], " +
                     f"pos3=[{self.features['pos3'].min()}-{self.features['pos3'].max()}]")
        report.append(f"- 唯一配置数: {len(self.data[['pos1', 'pos2', 'pos3']].drop_duplicates())}种")

    def _add_independent_effects_report(self, report):
        if 'independent_effects' not in self.results:
            return

        report.append("\n## 🎯 位置独立效应分析\n")
        report.append("### 📝 分析说明")
        report.append("位置独立效应分析旨在识别每个配置位置对游戏指标的单独影响，")
        report.append("控制其他变量的情况下，观察单个位置数值变化时指标的边际变化。")
        report.append("边际效应表示该位置数值每增加1个单位时，目标指标的平均变化量。\n")

        for pos in ['pos1', 'pos2', 'pos3']:
            if pos in self.results['independent_effects']:
                effects = self.results['independent_effects'][pos]
                report.append(f"### {pos}的独立效应:")

                for metric, data in effects.items():
                    if 'avg_marginal_diff' in data:
                        avg_diff = data['avg_marginal_diff']
                        volatility = data['marginal_volatility']
                        report.append(f"- **{metric}**: 平均边际效应 {avg_diff:.3f}, 波动性 {volatility:.3f}")

    def _add_interaction_effects_report(self, report):
        if 'interaction_effects' not in self.results:
            return

        report.append("\n## 🔄 位置交互效应分析\n")
        report.append("### 📝 分析说明")
        report.append("交互效应分析探讨不同配置位置之间的相互作用。当两个位置的组合效果")
        report.append("超过各自独立效果的简单相加时，就存在交互效应。正的交互增益表示")
        report.append("两个位置协同作用，负的交互增益表示两个位置相互抵消。\n")

        for pair, effects in self.results['interaction_effects'].items():
            report.append(f"### {pair}交互效应:")
            for metric, data in effects.items():
                gain = data['interaction_gain']
                strength = data['interaction_strength']
                report.append(f"- **{metric}**: 交互增益 {gain:.4f}, 效应强度 {strength:.4f}")

    def _add_dynamic_effects_report(self, report):
        if 'dynamic_effects' not in self.results:
            return

        report.append("\n## ⏱️ 动态影响分析\n")

        for pos, dynamics in self.results['dynamic_effects'].items():
            report.append(f"### {pos}的时序影响:")
            for stage, corr in dynamics.items():
                stage_name = {'early_correlation': '前期', 'middle_correlation': '中期', 'late_correlation': '后期'}.get(stage, stage)
                report.append(f"- **{stage_name}**: 相关性 {corr:.3f}")

    def _add_difficulty_position_effects_report(self, report):
        """添加DifficultyPosition影响分析报告 - 增强版"""
        if 'difficulty_position_effects' not in self.results:
            return

        report.append("\n## 🏆 DifficultyPosition影响分析 (增强版)\n")
        report.append("### 📝 指标说明与重要性")
        report.append("**DifficultyPosition** 是游戏体验设计中的核心指标，反映难点在游戏流程中的出现位置。")
        report.append("该指标直接影响玩家的体验节奏、挫败感和成就感的分布：")
        report.append("")
        report.append("**数值含义深度解析：**")
        report.append("- **0.00-0.25**: 前期难点 - 游戏前25%阶段出现难点")
        report.append("- **0.25-0.50**: 前中期难点 - 游戏25%-50%阶段出现难点")
        report.append("- **0.50-0.75**: 后中期难点 - 游戏50%-75%阶段出现难点")
        report.append("- **0.75-1.00**: 后期难点 - 游戏后25%阶段出现难点")
        report.append("- **1.00**: 无明显难点 - 体验平滑，无明显困难点")
        report.append("")
        report.append("**体验设计原则：** 四阶段均匀分布，每个阶段占25%的游戏进程。")
        report.append("")

        # 计算全局统计
        all_positions = []
        for pos in ['pos1', 'pos2', 'pos3']:
            if pos in self.results['difficulty_position_effects'] and 'value_effects' in self.results['difficulty_position_effects'][pos]:
                for value, stats in self.results['difficulty_position_effects'][pos]['value_effects'].items():
                    all_positions.extend([stats['mean']] * stats['count'])

        if all_positions:
            global_mean = np.mean(all_positions)
            global_std = np.std(all_positions)
            stage1_ratio = sum(1 for p in all_positions if p < 0.25) / len(all_positions)
            stage2_ratio = sum(1 for p in all_positions if 0.25 <= p < 0.50) / len(all_positions)
            stage3_ratio = sum(1 for p in all_positions if 0.50 <= p < 0.75) / len(all_positions)
            stage4_ratio = sum(1 for p in all_positions if 0.75 <= p < 1.0) / len(all_positions)
            perfect_ratio = sum(1 for p in all_positions if abs(p - 1.0) < 0.01) / len(all_positions)

            report.append("### 📊 全局DifficultyPosition分布特征")
            report.append(f"- **整体均值**: {global_mean:.3f}")
            report.append(f"- **标准差**: {global_std:.3f}")
            report.append(f"- **前期难点占比 (0.00-0.25)**: {stage1_ratio:.1%}")
            report.append(f"- **前中期难点占比 (0.25-0.50)**: {stage2_ratio:.1%}")
            report.append(f"- **后中期难点占比 (0.50-0.75)**: {stage3_ratio:.1%}")
            report.append(f"- **后期难点占比 (0.75-1.00)**: {stage4_ratio:.1%}")
            report.append(f"- **无难点占比**: {perfect_ratio:.1%}")
            report.append("")

        # 详细的位置影响分析
        for pos in ['pos1', 'pos2', 'pos3']:
            if pos in self.results['difficulty_position_effects']:
                pos_data = self.results['difficulty_position_effects'][pos]
                report.append(f"### 🎯 {pos.upper()}位置的DifficultyPosition影响详析")

                # 相关性分析
                if 'correlation' in pos_data:
                    corr_data = pos_data['correlation']
                    significance = "统计显著" if corr_data['significant'] else "统计不显著"
                    corr_strength = "强相关" if abs(corr_data['correlation']) > 0.5 else "中等相关" if abs(corr_data['correlation']) > 0.3 else "弱相关"
                    corr_direction = "正相关(数值越大难点越靠后)" if corr_data['correlation'] > 0 else "负相关(数值越大难点越靠前)"

                    report.append(f"**总体相关性分析：**")
                    report.append(f"- 相关系数: {corr_data['correlation']:.3f} ({corr_strength}，{corr_direction})")
                    report.append(f"- 显著性: p={corr_data['p_value']:.3f} ({significance})")
                    report.append("")

                # 详细的数值效应分析
                if 'value_effects' in pos_data:
                    value_effects = pos_data['value_effects']

                    report.append(f"**{pos.upper()}各数值配置的DifficultyPosition影响详表：**")
                    report.append("| 配置值 | 平均位置 | 标准差 | 中位数 | 样本数 | 位置特征 | 体验评价 |")
                    report.append("|--------|----------|--------|--------|--------|----------|----------|")

                    for value in sorted(value_effects.keys()):
                        stats = value_effects[value]
                        mean_pos = stats['mean']

                        # 位置特征描述
                        if mean_pos < 0.25:
                            pos_feature = "前期难点"
                            experience_eval = "Stage 1"
                        elif mean_pos < 0.50:
                            pos_feature = "前中期难点"
                            experience_eval = "Stage 2"
                        elif mean_pos < 0.75:
                            pos_feature = "后中期难点"
                            experience_eval = "Stage 3"
                        elif mean_pos < 1.0:
                            pos_feature = "后期难点"
                            experience_eval = "Stage 4"
                        else:
                            pos_feature = "无明显难点"
                            experience_eval = "No Difficulty"

                        report.append(f"| {value} | {mean_pos:.3f} | {stats['std']:.3f} | {stats['median']:.3f} | {stats['count']} | {pos_feature} | {experience_eval} |")

                    report.append("")

                # 配置推荐分析 - 基于数据特征而非简单推荐/不推荐
                if 'value_effects' in pos_data:
                    # 按照DifficultyPosition进行排序和分类
                    sorted_configs = sorted(value_effects.items(), key=lambda x: x[1]['mean'])

                    report.append(f"**{pos.upper()}配置效果排序分析 (按DifficultyPosition从早到晚)：**")

                    for i, (value, stats) in enumerate(sorted_configs):
                        rank = i + 1
                        mean_pos = stats['mean']
                        sample_count = stats['count']

                        # 详细的特征描述
                        if mean_pos < 0.25:
                            characteristic = f"使难点在前期出现(位置{mean_pos:.3f})"
                        elif mean_pos < 0.50:
                            characteristic = f"使难点在前中期出现(位置{mean_pos:.3f})"
                        elif mean_pos < 0.75:
                            characteristic = f"使难点在后中期出现(位置{mean_pos:.3f})"
                        elif mean_pos < 1.0:
                            characteristic = f"使难点在后期出现(位置{mean_pos:.3f})"
                        else:
                            characteristic = f"使关卡缺乏明显难点(位置{mean_pos:.3f})"

                        confidence = "高置信度" if sample_count > 100 else "中置信度" if sample_count > 30 else "低置信度"

                        report.append(f"{rank}. **配置值{value}**: {characteristic}")
                        report.append(f"   - 数据置信度: {confidence} (样本数:{sample_count})")
                        report.append("")

                # 配置分类统计 - 增强版
                categories = [
                    ('no_difficulty', '无明显难点配置', '体验平滑但可能缺乏挑战'),
                    ('early_difficulty', '前期难点配置', '开局挑战型，需注意挫败感控制'),
                    ('mid_difficulty', '中期难点配置', '渐进式挑战，推荐的体验节奏'),
                    ('late_difficulty', '后期难点配置', '后发制人型，适合构建层次感')
                ]

                report.append(f"**{pos.upper()}配置类型分布统计：**")
                for category_key, category_name, category_desc in categories:
                    config_key = f'{category_key}_configs'
                    if config_key in pos_data:
                        config_info = pos_data[config_key]
                        values_str = ', '.join(map(str, sorted(config_info['values'])))
                        avg_pos = config_info['avg_position']
                        count = config_info['count']
                        proportion = count / len(value_effects) if value_effects else 0

                        report.append(f"- **{category_name}** ({proportion:.1%}): 配置值[{values_str}]")
                        report.append(f"  - 平均DifficultyPosition: {avg_pos:.3f}")
                        report.append(f"  - 设计特点: {category_desc}")
                        report.append("")

        # 交互效应分析 - 增强版
        if 'interaction_effects' in self.results['difficulty_position_effects']:
            interaction_data = self.results['difficulty_position_effects']['interaction_effects']
            if interaction_data:
                report.append("### 🔄 位置间交互效应对DifficultyPosition的影响")
                report.append("**交互效应说明**: 衡量不同位置配置组合时产生的协同或冲突效应，正值表示协同增强，负值表示相互抵消。")
                report.append("")

                sorted_interactions = sorted(interaction_data.items(), key=lambda x: x[1]['interaction_gain'], reverse=True)

                for pair, data in sorted_interactions:
                    gain = data['interaction_gain']
                    r2_base = data['r2_base']
                    r2_inter = data['r2_interaction']
                    sample_count = data['sample_count']

                    if gain > 0.01:
                        effect_desc = "显著协同效应"
                        impact_desc = "位置间配置产生叠加增强"
                    elif gain > 0.005:
                        effect_desc = "轻微协同效应"
                        impact_desc = "位置间配置略有协同"
                    elif gain > -0.005:
                        effect_desc = "无明显交互"
                        impact_desc = "位置间相对独立"
                    elif gain > -0.01:
                        effect_desc = "轻微冲突效应"
                        impact_desc = "位置间配置略有抵消"
                    else:
                        effect_desc = "显著冲突效应"
                        impact_desc = "位置间配置相互干扰"

                    report.append(f"**{pair}交互分析:**")
                    report.append(f"- 交互增益: {gain:.4f} ({effect_desc})")
                    report.append(f"- 模型改进: {r2_base:.3f} → {r2_inter:.3f}")
                    report.append(f"- 效应解释: {impact_desc}")
                    report.append(f"- 样本规模: {sample_count}")
                    report.append("")


    def _add_mechanism_effects_report(self, report):
        if 'mechanism_effects' not in self.results:
            return

        report.append("\n## 🔍 影响机制分析 (增强版)\n")
        report.append("### 📝 机制分析理论基础")
        report.append("**影响机制分析** 是一种深度解析体验配置如何产生最终效果的方法论。该分析基于中介效应理论，")
        report.append("旨在识别配置影响游戏指标的具体路径和中间环节。")
        report.append("")
        report.append("**核心概念说明：**")
        report.append("- **直接效应(Direct Effect)**: 配置位置直接对目标指标产生的影响，不通过其他中间变量")
        report.append("- **中介效应(Mediation Effect)**: 配置通过影响中间变量，再由中间变量影响目标指标的间接路径")
        report.append("- **总效应(Total Effect)**: 直接效应 + 所有中介效应的总和")
        report.append("- **中介路径强度**: 衡量特定中介路径对总影响的贡献度，数值越大贡献越大")
        report.append("")
        report.append("**实践价值：**")
        report.append("- 理解WHY：不仅知道\"配置X影响指标Y\"，还知道\"是通过什么机制影响的\"")
        report.append("- 精准优化：针对具体机制进行优化，而非盲目调整配置")
        report.append("- 副作用预测：了解调整某个配置可能对其他指标产生的连锁影响")
        report.append("")

        for pos, mechanism in self.results['mechanism_effects'].items():
            report.append(f"### 🎯 {pos.upper()}位置的影响机制深度解析")

            if 'direct_effect' in mechanism:
                direct_effect = mechanism['direct_effect']
                effect_strength = "强直接影响" if abs(direct_effect) > 0.3 else "中等直接影响" if abs(direct_effect) > 0.1 else "弱直接影响"
                effect_direction = "正向推动" if direct_effect > 0 else "负向抑制"

                report.append(f"**直接效应分析:**")
                report.append(f"- 直接效应系数: {direct_effect:.3f} ({effect_strength}，{effect_direction})")
                report.append(f"- 效应解释: {pos.upper()}配置每增加1个单位，目标指标{('增加' if direct_effect > 0 else '减少')}{abs(direct_effect):.3f}个单位")

                if abs(direct_effect) > 0.2:
                    report.append(f"- **设计建议**: 该位置对目标指标有显著直接影响，是关键调节点")
                elif abs(direct_effect) > 0.1:
                    report.append(f"- **设计建议**: 该位置有中等程度的直接影响，可作为微调参数")
                else:
                    report.append(f"- **设计建议**: 该位置直接影响较小，主要通过中介机制发挥作用")
                report.append("")

            if 'strongest_mediation_path' in mechanism:
                strongest = mechanism['strongest_mediation_path']
                mediator = strongest['mediator']
                strength = strongest['strength']

                # 中介变量含义解释
                mediator_explanations = {
                    'PeakDockCount': '游戏过程中Dock区域的最大瓦片数量，反映游戏过程的复杂程度和压力峰值',
                    'PressureValueMean': '游戏全程压力值的平均水平，衡量整体游戏难度和压迫感',
                    'PressureValueMax': '游戏过程中的最大压力值，反映游戏的难度峰值',
                    'PressureValueStdDev': '压力值的标准差，衡量游戏难度波动的剧烈程度',
                    'InitialMinCost': '游戏初期的最小消除成本，反映开局的难易程度',
                    'FinalDifficulty': '游戏最终难度评分，综合衡量关卡的整体挑战水平'
                }

                mediator_desc = mediator_explanations.get(mediator, f'{mediator}相关指标')
                strength_level = "显著中介效应" if abs(strength) > 0.2 else "中等中介效应" if abs(strength) > 0.1 else "轻微中介效应"

                report.append(f"**最强中介路径分析:**")
                report.append(f"- 中介变量: {mediator}")
                report.append(f"- 中介含义: {mediator_desc}")
                report.append(f"- 中介强度: {strength:.3f} ({strength_level})")
                report.append(f"- **作用机制**: {pos.upper()}配置 → 影响{mediator} → {mediator}影响目标指标")

                if abs(strength) > 0.15:
                    report.append(f"- **优化策略**: 该中介路径是主要影响机制，调整{pos.upper()}时需重点关注对{mediator}的影响")
                else:
                    report.append(f"- **优化策略**: 该中介路径影响相对较小，可作为辅助优化方向")

                # 添加具体的优化建议
                if mediator == 'PeakDockCount':
                    report.append(f"- **具体建议**: 通过调整{pos.upper()}来控制Dock区域压力，避免瓦片堆积过多")
                elif mediator == 'PressureValueMean':
                    report.append(f"- **具体建议**: 通过{pos.upper()}调节整体游戏压力水平，保持挑战与体验的平衡")
                elif mediator == 'DifficultyPosition':
                    report.append(f"- **具体建议**: 通过{pos.upper()}控制难点出现时机，优化游戏节奏")

                report.append("")

            # 添加综合机制评估
            if 'direct_effect' in mechanism and 'strongest_mediation_path' in mechanism:
                direct = abs(mechanism['direct_effect'])
                indirect = abs(mechanism['strongest_mediation_path']['strength'])

                if direct > indirect * 1.5:
                    mechanism_type = "直接主导型"
                    mechanism_desc = f"该位置主要通过直接效应影响目标指标，中介效应相对较弱"
                    optimization_focus = "重点关注该位置的直接调节效果"
                elif indirect > direct * 1.5:
                    mechanism_type = "中介主导型"
                    mechanism_desc = f"该位置主要通过中介机制影响目标指标，直接效应相对较小"
                    optimization_focus = f"重点关注{mechanism['strongest_mediation_path']['mediator']}的变化"
                else:
                    mechanism_type = "混合影响型"
                    mechanism_desc = f"该位置通过直接和中介两种机制共同影响目标指标"
                    optimization_focus = "需要同时考虑直接效应和中介效应"

                report.append(f"**综合机制评估:**")
                report.append(f"- 机制类型: {mechanism_type}")
                report.append(f"- 机制特征: {mechanism_desc}")
                report.append(f"- 优化重点: {optimization_focus}")
                report.append("")

        report.append("---")
        report.append("💡 **机制分析应用指南:**")
        report.append("1. **直接主导型位置**: 直接调整配置值，效果立竿见影")
        report.append("2. **中介主导型位置**: 关注中介变量变化，通过间接路径优化")
        report.append("3. **混合影响型位置**: 综合考虑直接和间接效应，全面评估调整影响")
        report.append("4. **机制验证**: 建议通过A/B测试验证识别出的影响机制")
        report.append("")

    def _add_key_findings_and_recommendations(self, report):
        report.append("\n## 💡 关键发现与建议\n")
        report.append("### 📝 结果解读说明")
        report.append("- **位置重要性排序**: 基于各位置对主要指标的平均绝对相关性排序")
        report.append("- **交互效应强度**: 衡量位置间协同或冲突的程度，绝对值越大影响越明显")
        report.append("- **中介效应**: 分析位置通过其他指标间接影响目标指标的路径")
        report.append("- **配置建议**: 基于统计分析结果提供的优化方向，需结合实际游戏体验验证\n")

        # 位置重要性排序
        if 'correlations' in self.results:
            pos_importance = {}
            for pos in ['pos1', 'pos2', 'pos3']:
                avg_abs_corr = np.mean([
                    abs(self.results['correlations']['position_correlations'][pos][m]['correlation'])
                    for m in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']
                    if m in self.results['correlations']['position_correlations'][pos]
                ])
                pos_importance[pos] = avg_abs_corr

            sorted_positions = sorted(pos_importance.items(), key=lambda x: x[1], reverse=True)
            report.append("### 主要发现:")
            report.append(f"1. **位置重要性排序**: {' > '.join([f'{p}({v:.3f})' for p, v in sorted_positions])}")

        # 关键交互效应
        if 'interaction_effects' in self.results:
            max_interaction = None
            max_strength = 0
            for pair, effects in self.results['interaction_effects'].items():
                for metric, data in effects.items():
                    if data['interaction_strength'] > max_strength:
                        max_strength = data['interaction_strength']
                        max_interaction = f"{pair}->{metric}"
            if max_interaction:
                report.append(f"2. **最关键交互效应**: {max_interaction} (strength: {max_strength:.3f})")

        report.append("\n### 优化建议:")
        report.append("1. 重点关注影响力最大的位置参数")
        report.append("2. 考虑位置间的交互效应，避免单纯的独立调整")
        report.append("3. 根据中介机制针对性优化，提高调整精度")
        report.append("4. **DifficultyPosition优化建议**：")
        report.append("   - **理想范围**: 0.4-0.7（中期难点），提供良好的挑战节奏")
        report.append("   - **避免**: 0-0.3（前期难点），容易造成开局挫败")
        report.append("   - **可接受**: 0.7-0.99（后期难点），渐进式挑战")
        report.append("   - **特殊情况**: DifficultyPosition=1（无明显难点）适合休闲体验")
        report.append("5. 建议结合A/B测试验证统计分析结果")

        report.append("\n### ⚠️ 注意事项:")
        report.append("- 本分析基于历史数据，结果需要在实际环境中验证")
        report.append("- 统计显著性不等于实际显著性，需要结合游戏设计目标判断")
        report.append("- 配置调整应该渐进式进行，避免大幅变动影响玩家体验")
        report.append("- 建议定期重新分析以适应游戏发展和玩家行为变化")

    def _add_value_specific_report(self, report):
        """删除数值特异性分析 - 已移除"""
        pass

    def _add_gradient_effects_report(self, report):
        """添加数值梯度效应报告"""
        if 'gradient_effects' not in self.results:
            return

        report.append("\n## 📈 数值梯度效应分析\n")
        report.append("### 📝 梯度效应分析说明")
        report.append("**数值梯度效应分析** 研究体验配置数值从低到高变化时对游戏指标产生的渐进性影响。")
        report.append("通过分析数值变化的趋势和模式，识别配置的敏感区间和最优取值范围。")
        report.append("")
        report.append("**核心概念：**")
        report.append("- **梯度斜率**: 数值每增加1个单位时指标的平均变化量")
        report.append("- **敏感区间**: 梯度变化剧烈的数值范围，小幅调整产生显著影响")
        report.append("- **平稳区间**: 梯度变化平缓的数值范围，调整影响相对稳定")
        report.append("- **拐点**: 梯度方向发生明显变化的关键数值点")
        report.append("")
        report.append("**分析价值：**")
        report.append("- **精确调优**: 识别数值调整的最佳方向和幅度")
        report.append("- **风险控制**: 避免在敏感区间进行大幅调整")
        report.append("- **效果预测**: 基于梯度趋势预测数值变化的影响")
        report.append("")

        for pos in ['pos1', 'pos2', 'pos3']:
            if pos in self.results['gradient_effects']:
                report.append(f"### {pos}梯度效应:")
                pos_gradients = self.results['gradient_effects'][pos]

                for metric in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']:
                    if metric in pos_gradients:
                        gradient_data = pos_gradients[metric]
                        critical_point = gradient_data['critical_point']
                        max_gradient = gradient_data['max_gradient']
                        avg_gradient = gradient_data['avg_gradient']

                        report.append(f"- **{metric}梯度**: 平均梯度{avg_gradient:.3f}, 最大梯度{max_gradient:.3f}")
                        if critical_point:
                            report.append(f"  - 关键转折点: 数值{critical_point}")

                report.append("")

    def _add_dock_sequence_report(self, report):
        """删除Dock序列深度分析 - 已移除"""
        pass

    def _add_pressure_dynamics_report(self, report):
        """添加压力动态分析报告"""
        if 'pressure_dynamics' not in self.results:
            return

        report.append("\n## ⚡ 压力动态分析\n")

        pressure_names = {
            'PressureValueMean': '平均压力',
            'PressureValueMax': '峰值压力',
            'PressureValueStdDev': '压力波动'
        }

        for pos in ['pos1', 'pos2', 'pos3']:
            if pos in self.results['pressure_dynamics']:
                report.append(f"### {pos}压力动态:")
                pos_data = self.results['pressure_dynamics'][pos]

                # 分析每种压力类型
                for pressure_type, pressure_name in pressure_names.items():
                    report.append(f"#### {pressure_name}:")

                    # 找出极值配置
                    min_pressure_config = None
                    max_pressure_config = None
                    min_pressure = float('inf')
                    max_pressure = 0

                    for value in range(1, 10):
                        value_key = f'value_{value}'
                        if (value_key in pos_data and
                            pressure_type in pos_data[value_key] and
                            'statistics' in pos_data[value_key][pressure_type]):

                            stats = pos_data[value_key][pressure_type]['statistics']
                            mean_pressure = stats['mean']

                            if mean_pressure < min_pressure:
                                min_pressure = mean_pressure
                                min_pressure_config = value
                            if mean_pressure > max_pressure:
                                max_pressure = mean_pressure
                                max_pressure_config = value

                    if min_pressure_config:
                        report.append(f"- **最低压力**: 数值{min_pressure_config}, {pressure_name}{min_pressure:.3f}")
                    if max_pressure_config:
                        report.append(f"- **最高压力**: 数值{max_pressure_config}, {pressure_name}{max_pressure:.3f}")

                report.append("")

    def _print_key_findings(self):
        """输出关键发现摘要"""
        print("\n💡 === 关键发现摘要 ===")

        # 位置重要性排序
        if 'correlations' in self.results:
            pos_importance = {}
            for pos in ['pos1', 'pos2', 'pos3']:
                avg_abs_corr = np.mean([
                    abs(self.results['correlations']['position_correlations'][pos][m]['correlation'])
                    for m in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']
                    if m in self.results['correlations']['position_correlations'][pos]
                ])
                pos_importance[pos] = avg_abs_corr

            sorted_positions = sorted(pos_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"🎯 位置重要性排序: {' > '.join([f'{p}({v:.3f})' for p, v in sorted_positions])}")

        # 最强交互效应
        if 'interaction_effects' in self.results:
            max_interaction = None
            max_strength = 0
            for pair, effects in self.results['interaction_effects'].items():
                for metric, data in effects.items():
                    if data['interaction_strength'] > max_strength:
                        max_strength = data['interaction_strength']
                        max_interaction = f"{pair}->{metric}"
            if max_interaction:
                print(f"🔄 最强交互效应: {max_interaction} (strength: {max_strength:.3f})")

        # 最关键中介路径
        if 'mechanism_effects' in self.results:
            strongest_mediation = None
            max_mediation_strength = 0
            for pos, mechanism in self.results['mechanism_effects'].items():
                if 'strongest_mediation_path' in mechanism:
                    strength = abs(mechanism['strongest_mediation_path']['strength'])
                    if strength > max_mediation_strength:
                        max_mediation_strength = strength
                        strongest_mediation = f"{pos}->{mechanism['strongest_mediation_path']['mediator']}->DifficultyScore"
            if strongest_mediation:
                print(f"🔍 最关键中介路径: {strongest_mediation} (strength: {max_mediation_strength:.3f})")

    # 基础可视化方法（简化版实现）
    def _plot_position_correlation_heatmap(self, output_path: Path):
        """绘制位置相关性热力图"""
        if 'correlations' not in self.results:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # 构建相关性矩阵
        positions = ['pos1', 'pos2', 'pos3']
        metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']
        corr_matrix = []

        for pos in positions:
            pos_corrs = []
            for metric in metrics:
                if pos in self.results['correlations']['position_correlations'] and \
                   metric in self.results['correlations']['position_correlations'][pos]:
                    corr = self.results['correlations']['position_correlations'][pos][metric]['correlation']
                    pos_corrs.append(corr)
                else:
                    pos_corrs.append(0)
            corr_matrix.append(pos_corrs)

        corr_matrix = np.array(corr_matrix)

        # 绘制热力图
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # 设置标签
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions)

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='相关系数')

        # 添加数值标签
        for i in range(len(positions)):
            for j in range(len(metrics)):
                ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                       ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        plt.title('Position-Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_path / 'position_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pattern_boxplots(self, output_path: Path):
        """绘制配置模式箱线图"""
        if 'patterns' not in self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        patterns = self.results['patterns']['sequence_patterns']

        metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']

        for idx, metric in enumerate(metrics):
            ax = axes[idx//2, idx%2]
            pattern_names = []
            pattern_means = []
            pattern_stds = []

            for pattern_name, pattern_data in patterns.items():
                if metric in pattern_data and 'mean' in pattern_data[metric]:
                    pattern_names.append(pattern_name)
                    pattern_means.append(pattern_data[metric]['mean'])
                    pattern_stds.append(pattern_data[metric].get('std', 0))

            if pattern_names:
                bars = ax.bar(pattern_names, pattern_means, yerr=pattern_stds, alpha=0.7, capsize=5)
                ax.set_title(f'{metric} Configuration Pattern Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)

                # 添加数值标签
                for bar, mean in zip(bars, pattern_means):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                           f'{mean:.2f}', ha='center', va='bottom')

        plt.suptitle('配置模式分析箱线图')
        plt.tight_layout()
        plt.savefig(output_path / 'pattern_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, output_path: Path):
        """绘制特征重要性图"""
        if 'models' not in self.results:
            return

        key_metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(18, 6))

        for i, metric in enumerate(key_metrics):
            if metric in self.results['models']:
                importance = self.results['models'][metric]['feature_importance']

                # 取前8个重要特征
                top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])

                features = list(top_features.keys())
                values = list(top_features.values())

                bars = axes[i].barh(features, values, alpha=0.7)
                axes[i].set_title(f'{metric}\nFeature Importance (R²={self.results["models"][metric]["r2_score"]:.3f})')
                axes[i].set_xlabel('Importance Score')

                # 颜色映射
                colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

        plt.suptitle('机器学习模型特征重要性分析')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_config_distribution(self, output_path: Path):
        """绘制配置分布散点图，增加数值安全检查"""
        try:
            # 检查必要数据
            if len(self.features) == 0 or len(self.data) == 0:
                print("   ⚠️ 无配置分布数据，跳过绘制")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 3D配置空间投影 (pos1 vs pos2，颜色表示pos3)
            pos1_vals = self.features['pos1'].values
            pos2_vals = self.features['pos2'].values
            pos3_vals = self.features['pos3'].values

            # 数值范围检查
            if np.any(~np.isfinite(pos1_vals)) or np.any(~np.isfinite(pos2_vals)) or np.any(~np.isfinite(pos3_vals)):
                print("   ⚠️ 配置数据包含无效值，使用有效子集")
                valid_mask = np.isfinite(pos1_vals) & np.isfinite(pos2_vals) & np.isfinite(pos3_vals)
                pos1_vals = pos1_vals[valid_mask]
                pos2_vals = pos2_vals[valid_mask]
                pos3_vals = pos3_vals[valid_mask]

            if len(pos1_vals) > 0:
                scatter = axes[0,0].scatter(pos1_vals, pos2_vals, c=pos3_vals,
                                          cmap='viridis', alpha=0.6, s=1)
                axes[0,0].set_xlabel('Position 1 Value')
                axes[0,0].set_ylabel('Position 2 Value')
                axes[0,0].set_title('Configuration Space Distribution (Color=pos3)')
                plt.colorbar(scatter, ax=axes[0,0], label='pos3 Value')

            # 配置与DifficultyScore的关系
            if 'DifficultyScore' in self.data.columns:
                config_sum = self.features['config_sum'].values
                difficulty = self.data['DifficultyScore'].values

                # 过滤无效值
                valid_mask = np.isfinite(config_sum) & np.isfinite(difficulty)
                if np.any(valid_mask):
                    axes[0,1].scatter(config_sum[valid_mask], difficulty[valid_mask], alpha=0.3, s=1)
                    axes[0,1].set_xlabel('Configuration Sum')
                    axes[0,1].set_ylabel('DifficultyScore')
                    axes[0,1].set_title('Configuration Sum vs Difficulty Score')

            # 配置标准差分布
            config_std = self.features['config_std'].values
            valid_std = config_std[np.isfinite(config_std)]
            if len(valid_std) > 0:
                axes[1,0].hist(valid_std, bins=min(30, len(valid_std)//10), alpha=0.7, edgecolor='black')
                axes[1,0].set_xlabel('Configuration Standard Deviation')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].set_title('Configuration Standard Deviation Distribution')

            # 极值配置效应分析
            if 'DifficultyScore' in self.data.columns:
                extreme_low = self.data[self.features['has_extreme_low'] == 1]
                extreme_high = self.data[self.features['has_extreme_high'] == 1]
                normal = self.data[(self.features['has_extreme_low'] == 0) & (self.features['has_extreme_high'] == 0)]

                if len(extreme_low) > 0 and len(extreme_high) > 0 and len(normal) > 0:
                    means = []
                    categories = []

                    for name, group in [('极低配置', extreme_low), ('正常配置', normal), ('极高配置', extreme_high)]:
                        difficulty_vals = group['DifficultyScore'].values
                        valid_vals = difficulty_vals[np.isfinite(difficulty_vals)]
                        if len(valid_vals) > 0:
                            categories.append(name)
                            means.append(np.mean(valid_vals))

                    if means:
                        colors = ['red', 'gray', 'blue'][:len(means)]
                        bars = axes[1,1].bar(categories, means, alpha=0.7, color=colors)
                        axes[1,1].set_ylabel('Average DifficultyScore')
                        axes[1,1].set_title('Extreme Configuration Effect Analysis')

                        # 添加数值标签
                        for bar, mean in zip(bars, means):
                            if np.isfinite(mean):
                                axes[1,1].text(bar.get_x() + bar.get_width()/2,
                                              bar.get_height() + bar.get_height()*0.01,
                                              f'{mean:.2f}', ha='center', va='bottom')

            plt.suptitle('体验配置分布特征分析')
            plt.tight_layout()
            plt.savefig(output_path / 'config_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"   ❌ 配置分布图绘制错误: {str(e)}")
            plt.close('all')

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
        ax1.set_title('Prediction Model R² Performance Score')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Baseline(0.5)')
        ax1.legend()

        # 为每个bar添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # RMSE分数
        bars2 = ax2.bar(metrics, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Prediction Model RMSE Error')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)

        plt.suptitle('Machine Learning Model Performance Evaluation')
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_value_impact_heatmaps(self, output_path: Path):
        """绘制数值影响矩阵热力图 - 9×3矩阵显示每个数值在每个位置的影响"""
        if 'value_specific_effects' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        # 为每个指标创建热力图
        metrics = ['DifficultyScore', 'success_rate', 'PressureValueMean']
        metric_names = ['难度分数', '胜率', '平均压力']

        for idx, (metric_key, metric_name) in enumerate(zip(metrics, metric_names)):
            # 构建9x3矩阵
            matrix = np.zeros((9, 3))

            for value in range(1, 10):
                value_key = f"value_{value}"
                if value_key in self.results['value_specific_effects']:
                    for pos_idx, pos in enumerate(['pos1', 'pos2', 'pos3']):
                        if pos in self.results['value_specific_effects'][value_key]:
                            pos_data = self.results['value_specific_effects'][value_key][pos]

                            if metric_key == 'DifficultyScore' and 'difficulty_impact' in pos_data:
                                matrix[value-1, pos_idx] = pos_data['difficulty_impact']['mean']
                            elif metric_key == 'success_rate' and 'win_rate' in pos_data:
                                matrix[value-1, pos_idx] = pos_data['win_rate']['success_rate']
                            elif metric_key == 'PressureValueMean' and 'pressure_impact' in pos_data:
                                pressure_data = pos_data['pressure_impact']
                                if 'PressureValueMean' in pressure_data:
                                    matrix[value-1, pos_idx] = pressure_data['PressureValueMean']['mean']

            # 绘制热力图
            im = axes[idx].imshow(matrix, aspect='auto', cmap='RdYlBu_r')

            # 设置标签
            axes[idx].set_xticks(range(3))
            axes[idx].set_xticklabels(['Position1', 'Position2', 'Position3'])
            axes[idx].set_yticks(range(9))
            axes[idx].set_yticklabels([f'Value{i}' for i in range(1, 10)])
            axes[idx].set_title(f'{metric_name} Impact Matrix')

            # 添加颜色条
            plt.colorbar(im, ax=axes[idx])

        plt.suptitle('数值-位置影响矩阵热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'value_impact_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_gradient_curves(self, output_path: Path):
        """绘制数值梯度效应曲线"""
        if 'gradient_effects' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, pos in enumerate(['pos1', 'pos2', 'pos3']):
            if pos in self.results['gradient_effects']:
                pos_gradients = self.results['gradient_effects'][pos]

                for metric in ['DifficultyScore', 'PeakDockCount', 'PressureValueMean']:
                    if metric in pos_gradients:
                        data = pos_gradients[metric]
                        value_means = data['value_means']
                        critical_point = data['critical_point']

                        # 过滤掉nan值
                        valid_indices = [i for i, val in enumerate(value_means) if not np.isnan(val)]
                        valid_values = [i+1 for i in valid_indices]
                        valid_means = [value_means[i] for i in valid_indices]

                        if len(valid_values) > 2:
                            axes[idx].plot(valid_values, valid_means, 'o-', label=metric, linewidth=2, markersize=6)

                            # 标记临界点
                            if critical_point and critical_point in valid_values:
                                critical_idx = valid_values.index(critical_point)
                                axes[idx].scatter([critical_point], [valid_means[critical_idx]],
                                                s=100, c='red', marker='*', zorder=5)

                axes[idx].set_xlabel('Configuration Value')
                axes[idx].set_ylabel('Metric Mean')
                axes[idx].set_title(f'{pos} Gradient Effect Curve')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Value Gradient Effect Analysis Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'gradient_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_difficulty_position_analysis(self, output_path: Path):
        """绘制DifficultyPosition影响分析图"""
        if 'difficulty_position_effects' not in self.results:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 上排：每个位置对DifficultyPosition的影响
        for i, pos in enumerate(['pos1', 'pos2', 'pos3']):
            if pos in self.results['difficulty_position_effects']:
                pos_data = self.results['difficulty_position_effects'][pos]

                if 'value_effects' in pos_data and pos_data['value_effects']:
                    values = list(pos_data['value_effects'].keys())
                    means = [pos_data['value_effects'][v]['mean'] for v in values]
                    stds = [pos_data['value_effects'][v]['std'] for v in values]

                    bars = axes[0, i].bar(values, means, yerr=stds, alpha=0.7, capsize=5)
                    axes[0, i].set_xlabel(f'{pos} Value')
                    axes[0, i].set_ylabel('DifficultyPosition Average')
                    axes[0, i].set_title(f'{pos} Effect on DifficultyPosition')
                    axes[0, i].grid(True, alpha=0.3)

                    # 添加数值标签
                    for bar, mean in zip(bars, means):
                        axes[0, i].text(bar.get_x() + bar.get_width()/2,
                                       bar.get_height() + bar.get_height()*0.01,
                                       f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

        # 下排：交互效应分析
        if 'interaction_effects' in self.results['difficulty_position_effects']:
            interaction_data = self.results['difficulty_position_effects']['interaction_effects']

            if interaction_data:
                pairs = list(interaction_data.keys())
                gains = [interaction_data[pair]['interaction_gain'] for pair in pairs]

                # 合并下排三个子图为一个大图
                ax_combined = plt.subplot(2, 1, 2)
                bars = ax_combined.bar(pairs, gains, alpha=0.7,
                                     color=['red' if g > 0 else 'blue' for g in gains])
                ax_combined.set_xlabel('Position Combination')
                ax_combined.set_ylabel('Interaction Effect Gain')
                ax_combined.set_title('DifficultyPosition Interaction Effect Analysis')
                ax_combined.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax_combined.grid(True, alpha=0.3)

                # 添加数值标签
                for bar, gain in zip(bars, gains):
                    ax_combined.text(bar.get_x() + bar.get_width()/2,
                                   bar.get_height() + (0.01 if gain > 0 else -0.03),
                                   f'{gain:.4f}', ha='center',
                                   va='bottom' if gain > 0 else 'top', fontsize=9)

                # 隐藏下排原有的三个子图
                for i in range(3):
                    axes[1, i].set_visible(False)

        plt.suptitle('DifficultyPosition位置影响分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'difficulty_position_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_dock_sequence_patterns(self, output_path: Path):
        """绘制Dock序列模式图"""
        if 'dock_deep_analysis' not in self.results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 为每个位置绘制不同数值的成功率分析
        colors = plt.cm.Set3(np.linspace(0, 1, 9))

        for pos_idx, pos in enumerate(['pos1', 'pos2', 'pos3']):
            if pos in self.results['dock_deep_analysis']:
                pos_data = self.results['dock_deep_analysis'][pos]

                # 成功率分析
                success_rates = []
                values = []

                for value in range(1, 10):
                    value_key = f'value_{value}'
                    if value_key in pos_data:
                        success_rates.append(pos_data[value_key]['success_rate'])
                        values.append(value)

                if success_rates:
                    bars = axes[pos_idx].bar(values, success_rates, color=colors[:len(values)], alpha=0.7)
                    axes[pos_idx].set_xlabel('Configuration Value')
                    axes[pos_idx].set_ylabel('Success Rate')
                    axes[pos_idx].set_title(f'{pos} - Win Rate Performance')
                    axes[pos_idx].set_ylim(0, 1)

                    # 添加数值标签
                    for bar, rate in zip(bars, success_rates):
                        axes[pos_idx].text(bar.get_x() + bar.get_width()/2,
                                          bar.get_height() + 0.02,
                                          f'{rate:.3f}', ha='center', va='bottom')

        plt.suptitle('Dock序列模式与成功率分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'dock_sequence_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pressure_dynamics(self, output_path: Path):
        """绘制压力动态分布图"""
        if 'pressure_dynamics' not in self.results:
            return

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        pressure_types = ['PressureValueMean', 'PressureValueMax', 'PressureValueStdDev']
        pressure_names = ['平均压力', '最大压力', '压力波动']

        for row, (pressure_type, pressure_name) in enumerate(zip(pressure_types, pressure_names)):
            for col, pos in enumerate(['pos1', 'pos2', 'pos3']):
                if pos in self.results['pressure_dynamics']:
                    pos_data = self.results['pressure_dynamics'][pos]

                    # 收集数据
                    values = []
                    means = []
                    q95s = []

                    for value in range(1, 10):
                        value_key = f'value_{value}'
                        if (value_key in pos_data and
                            pressure_type in pos_data[value_key] and
                            'statistics' in pos_data[value_key][pressure_type]):

                            stats = pos_data[value_key][pressure_type]['statistics']
                            values.append(value)
                            means.append(stats['mean'])
                            q95s.append(stats['q95'])

                    if values:
                        # 绘制平均值线
                        axes[row, col].plot(values, means, 'o-', label='Mean', linewidth=2, markersize=6)

                        # 绘制95%分位数线
                        axes[row, col].plot(values, q95s, 's--', label='95% Percentile', alpha=0.7)

                        axes[row, col].set_xlabel('Configuration Value')
                        axes[row, col].set_ylabel(pressure_name)
                        axes[row, col].set_title(f'{pos} - {pressure_name} Dynamics')
                        axes[row, col].legend()
                        axes[row, col].grid(True, alpha=0.3)

        plt.suptitle('Pressure Metric Dynamics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'pressure_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_visualizations(self, output_dir: str = None):
        """兼容方法：创建可视化图表"""
        self.create_enhanced_visualizations(output_dir)

    def generate_report(self, output_path: str = None) -> str:
        """兼容方法：生成基础分析报告"""
        return self.generate_enhanced_report(output_path)


def main():
    """主函数 - 运行深度分析"""
    import sys
    import os

    # 设置UTF-8编码输出
    if sys.platform.startswith('win'):
        os.system('chcp 65001 >nul 2>&1')
        # 设置控制台输出编码
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("🚀 体验模式配置深度影响分析工具启动...", flush=True)

    try:
        # 创建分析器实例 - 默认使用多文件模式
        analyzer = ExperienceConfigAnalyzer()

        # 运行深度分析
        results = analyzer.run_complete_analysis()

        print(f"\n✨ 深度分析完成总结:")
        print(f"   📋 处理数据: {results['data_summary']['total_records']:,}条记录")
        print(f"   🎯 唯一配置: {results['data_summary']['unique_configs']}种")
        print(f"   📊 目标指标: {results['data_summary']['target_metrics']}个")
        print(f"   🤖 最佳模型R²: {results['analysis_results']['best_model_r2']:.3f}")

        analysis_status = results['analysis_results']
        print(f"\n🔍 分析模块状态:")
        print(f"   ✅ 基础分析: {'完成' if analysis_status['basic_analysis'] else '未完成'}")
        print(f"   ✅ 高级分析: {'完成' if analysis_status['advanced_analysis'] else '未完成'}")
        print(f"   ✅ 动态分析: {'完成' if analysis_status['dynamic_analysis'] else '未完成'}")
        print(f"   ✅ 机制分析: {'完成' if analysis_status['mechanism_analysis'] else '未完成'}")

        # 输出数据来源信息
        if hasattr(analyzer, 'csv_files') and analyzer.csv_files:
            print(f"\n📂 数据来源: {len(analyzer.csv_files)}个CSV文件")
            for i, csv_file in enumerate(analyzer.csv_files[:5]):  # 最多显示前5个
                print(f"   {i+1}. {Path(csv_file).name}")
            if len(analyzer.csv_files) > 5:
                print(f"   ... 还有{len(analyzer.csv_files)-5}个文件")

    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def analyze_specific_directory(directory_path: str):
    """分析指定目录的CSV文件"""
    print(f"🎯 分析指定目录: {directory_path}")

    try:
        # 创建分析器实例指定目录
        analyzer = ExperienceConfigAnalyzer(csv_directory=directory_path)

        # 运行深度分析
        results = analyzer.run_complete_analysis()

        return results

    except Exception as e:
        print(f"❌ 指定目录分析失败: {str(e)}")
        raise


def analyze_single_file(file_path: str):
    """分析单个CSV文件"""
    print(f"📄 分析单个文件: {file_path}")

    try:
        # 创建分析器实例指定文件
        analyzer = ExperienceConfigAnalyzer(csv_path=file_path)

        # 运行深度分析
        results = analyzer.run_complete_analysis()

        return results

    except Exception as e:
        print(f"❌ 单文件分析失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()