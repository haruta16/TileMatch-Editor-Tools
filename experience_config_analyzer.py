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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
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
            'TotalMoves', 'InitialMinCost', 'DifficultyPosition'
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
                print(f"🔍 检测到analysis_charts目录，包含{len(csv_files)}个CSV文件")
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
        print("📊 加载和预处理数据...")

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

        print(f"🔍 发现{len(self.csv_files)}个CSV文件，准备批量加载...")

        # 简单直接加载，一个文件一个文件处理
        data_list = []
        total_rows = 0

        for i, csv_file in enumerate(self.csv_files):
            print(f"   正在加载 {csv_file.name}... ({i+1}/{len(self.csv_files)})")

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
        print("🔗 合并所有数据...")
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

        print("🔧 解析体验模式配置...")

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

    def _clean_data(self):
        """数据清洗，增强边界条件检查"""
        initial_count = len(self.data)
        print(f"🧹 开始数据清洗，初始记录数: {initial_count}")

        # 保存原始索引用于特征数据同步
        original_indices = self.data.index.copy()

        # 检查GameCompleted列是否存在
        if 'GameCompleted' in self.data.columns:
            # 移除GameCompleted=False的记录
            completed_mask = (self.data['GameCompleted'] == True) | (self.data['GameCompleted'] == 'True')
            self.data = self.data[completed_mask].copy()
            print(f"   完成游戏过滤: {len(self.data)}条记录")
        else:
            print("   ⚠️ 未找到GameCompleted列，跳过完成状态过滤")

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
        print("🔗 执行基础相关性分析...")

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

    def position_independent_analysis(self) -> Dict:
        """位置独立影响分析 - 控制其他变量分析单个位置的纯净效应"""
        print("🎯 执行位置独立影响分析...")

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
        print("🔄 执行位置交互效应分析...")

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
        print("⏱️ 执行动态影响分析...")

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
        print("🔍 执行影响机制分析...")

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

        # 深度分析（新增）
        independent_effects = self.position_independent_analysis()
        interaction_effects = self.interaction_analysis()
        dynamic_effects = self.dynamic_impact_analysis()
        mechanism_effects = self.mechanism_analysis()

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
        print("📊 创建增强可视化图表...")

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
            ("位置独立效应图", self._plot_independent_effects),
            ("交互效应图", self._plot_interaction_effects),
            ("机制分析图", self._plot_mechanism_analysis)
        ]

        successful_charts = 0
        for chart_name, chart_func in charts:
            try:
                print(f"   绘制 {chart_name}...")
                chart_func(output_path)
                successful_charts += 1
                print(f"   ✅ {chart_name} 完成")
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
                        axes[i].set_title(f'{pos}对DifficultyScore的边际效应')
                        axes[i].set_xlabel(f'{pos}数值')
                        axes[i].set_ylabel('DifficultyScore均值')
                        axes[i].grid(True, alpha=0.3)

        plt.suptitle('位置独立效应分析', fontsize=16, fontweight='bold')
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

            ax.set_xlabel('位置组合')
            ax.set_ylabel('交互效应强度')
            ax.set_title('位置间交互效应分析')
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

                    axes[i].set_title(f'{pos}的中介效应分析')
                    axes[i].set_xlabel('中介效应强度')

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
        print("📋 生成深度分析报告...")

        if output_path is None:
            # 多文件模式使用csv_directory，单文件模式使用csv_path
            if self.csv_path:
                output_path = Path(self.csv_path).parent / "enhanced_analysis_report.md"
            else:
                output_path = Path(self.csv_directory) / "enhanced_analysis_report.md"

        report = []
        report.append("# 体验模式配置[x,y,z]深度影响分析报告\n")

        # 数据源信息
        if self.csv_path:
            report.append(f"**数据源**: {self.csv_path}")
        else:
            report.append(f"**数据源**: {self.csv_directory} ({len(self.csv_files)}个CSV文件)")

        report.append(f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**样本数量**: {len(self.data):,}条\n")

        # 数据概览
        self._add_data_overview(report)

        # 位置独立效应分析
        self._add_independent_effects_report(report)

        # 交互效应分析
        self._add_interaction_effects_report(report)

        # 动态影响分析
        self._add_dynamic_effects_report(report)

        # 机制分析
        self._add_mechanism_effects_report(report)

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

    def _add_mechanism_effects_report(self, report):
        if 'mechanism_effects' not in self.results:
            return

        report.append("\n## 🔍 影响机制分析\n")

        for pos, mechanism in self.results['mechanism_effects'].items():
            report.append(f"### {pos}的影响机制:")
            if 'direct_effect' in mechanism:
                report.append(f"- **直接效应**: {mechanism['direct_effect']:.3f}")

            if 'strongest_mediation_path' in mechanism:
                strongest = mechanism['strongest_mediation_path']
                report.append(f"- **最强中介路径**: {strongest['mediator']} (strength: {strongest['strength']:.3f})")

    def _add_key_findings_and_recommendations(self, report):
        report.append("\n## 💡 关键发现与建议\n")

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

        plt.title('位置-指标相关性热力图')
        plt.tight_layout()
        plt.savefig(output_path / 'position_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pattern_boxplots(self, output_path: Path):
        """绘制配置模式箱线图"""
        if 'patterns' not in self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        patterns = self.results['patterns']['sequence_patterns']

        metrics = ['DifficultyScore', 'PeakDockCount', 'PressureValueMean', 'TotalMoves']

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
                ax.set_title(f'{metric}的配置模式对比')
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
                axes[i].set_title(f'{metric}\n特征重要性 (R²={self.results["models"][metric]["r2_score"]:.3f})')
                axes[i].set_xlabel('重要性分数')

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
                axes[0,0].set_xlabel('位置1数值')
                axes[0,0].set_ylabel('位置2数值')
                axes[0,0].set_title('配置空间分布 (颜色=pos3)')
                plt.colorbar(scatter, ax=axes[0,0], label='pos3数值')

            # 配置与DifficultyScore的关系
            if 'DifficultyScore' in self.data.columns:
                config_sum = self.features['config_sum'].values
                difficulty = self.data['DifficultyScore'].values

                # 过滤无效值
                valid_mask = np.isfinite(config_sum) & np.isfinite(difficulty)
                if np.any(valid_mask):
                    axes[0,1].scatter(config_sum[valid_mask], difficulty[valid_mask], alpha=0.3, s=1)
                    axes[0,1].set_xlabel('配置总和')
                    axes[0,1].set_ylabel('DifficultyScore')
                    axes[0,1].set_title('配置总和 vs 难度分数')

            # 配置标准差分布
            config_std = self.features['config_std'].values
            valid_std = config_std[np.isfinite(config_std)]
            if len(valid_std) > 0:
                axes[1,0].hist(valid_std, bins=min(30, len(valid_std)//10), alpha=0.7, edgecolor='black')
                axes[1,0].set_xlabel('配置标准差')
                axes[1,0].set_ylabel('频次')
                axes[1,0].set_title('配置标准差分布')

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
                        axes[1,1].set_ylabel('平均DifficultyScore')
                        axes[1,1].set_title('极值配置效应分析')

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
        ax1.set_title('预测模型R²性能评分')
        ax1.set_ylabel('R²分数')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='基准线(0.5)')
        ax1.legend()

        # 为每个bar添加数值标签
        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # RMSE分数
        bars2 = ax2.bar(metrics, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('预测模型RMSE误差')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)

        plt.suptitle('机器学习模型性能评估')
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
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

    print("🚀 体验模式配置深度影响分析工具启动...")

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