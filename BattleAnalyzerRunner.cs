// ========== 日志级别控制 ==========
// 通过定义符号控制详细日志输出程度
// Unity菜单: Edit → Project Settings → Player → Scripting Define Symbols
// 添加 VERBOSE_ANALYZER_LOGGING 启用详细日志（每个种子尝试都输出）
// 默认：NORMAL_ANALYZER_LOGGING（每100个任务输出进度）
#if !VERBOSE_ANALYZER_LOGGING
#define NORMAL_ANALYZER_LOGGING
#endif

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using DGuo.Client.TileMatch;
using DGuo.Client;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DGuo.Client.TileMatch.Analysis
{
    /// <summary>
    /// BattleAnalyzer自动运行器 - 读取all_level.csv配置，批量生成关卡并使用TileMatchBattleAnalyzerMgr自动运行
    /// </summary>
    public class BattleAnalyzerRunner
    {
        // ========== 常量定义 ==========

        /// <summary>
        /// 文件IO缓冲区大小：64KB
        /// </summary>
        private const int FILE_BUFFER_SIZE = 65536;

        /// <summary>
        /// CSV解析器StringBuilder初始容量
        /// </summary>
        private const int CSV_PARSER_BUFFER_SIZE = 256;
        /// <summary>
        /// CSV配置行数据 - 统一使用BatchLevelEvaluatorSimple的数据结构
        /// </summary>
        public class CsvLevelConfig
        {
            public int RowIndex { get; set; } // CSV行索引（从0开始，0=表头后第一行）
            public int TerrainId { get; set; }
            public int[] ExpFix1 { get; set; }
            public int[] ExpFix2 { get; set; }
            public int[] ExpFix3 { get; set; }
            public int[] ExpFix4 { get; set; }
            public int[] ExpFix5 { get; set; }
            public int[] ExpFix6 { get; set; }
            public int[] ExpRange1 { get; set; }
            public int TypeCount1 { get; set; }
            public int TypeCount2 { get; set; }
            public int TypeCount3 { get; set; }
            public int TypeCount4 { get; set; }
            public int TypeCount5 { get; set; }
            public int TypeCount6 { get; set; }
            public int TypeRange1 { get; set; }
            public (float min, float max)? PositionRange { get; set; }
            public (float min, float max)? ScoreRange { get; set; }
            public (int min, int max)? ConsecutiveLowPressureRange { get; set; }
            public (int min, int max)? TotalEarlyLowPressureRange { get; set; }
        }

        /// <summary>
        /// 自动游戏分析结果
        /// </summary>
        public class AnalysisResult
        {
            public string UniqueId { get; set; } // 唯一标识符
            public int RowIndex { get; set; } // CSV行索引
            public int TerrainId { get; set; }
            public string LevelName { get; set; }
            public string AlgorithmName { get; set; } // 生成算法版本名
            public int[] ExperienceMode { get; set; }
            public int ColorCount { get; set; }
            public int TotalTiles { get; set; }
            public int RandomSeed { get; set; } // 新增随机种子字段

            // 游戏执行结果
            public bool GameCompleted { get; set; }
            public int TotalMoves { get; set; }
            public int GameDurationMs { get; set; }
            public string CompletionStatus { get; set; }

            // BattleAnalyzer分析结果
            public int TotalAnalysisTimeMs { get; set; }
            public int SuccessfulGroups { get; set; }
            public List<int> TileIdSequence { get; set; } = new List<int>();
            public List<int> DockCountPerMove { get; set; } = new List<int>();

            // 关键快照数据
            public int PeakDockCount { get; set; }
            public int MinMovesToComplete { get; set; }
            public int InitialMinCost { get; set; } // 游戏开局时的最小cost值
            public double DifficultyPosition { get; set; } // 难点位置：0~1，表示peakdock在关卡进度中的位置
            public List<int> DockAfterTrioMatch { get; set; } = new List<int>();
            public List<int> SafeOptionCounts { get; set; } = new List<int>();
            public List<int> MinCostAfterTrioMatch { get; set; } = new List<int>();
            public List<int> MinCostOptionsAfterTrioMatch { get; set; } = new List<int>();
            public List<int> PressureValues { get; set; } = new List<int>(); // 压力值列表：开局+每次三消后

            // 压力值统计字段
            public double PressureValueMean { get; set; } // 压力值均值
            public int PressureValueMin { get; set; } // 压力值最小值
            public int PressureValueMax { get; set; } // 压力值最大值
            public double PressureValueStdDev { get; set; } // 压力值标准差
            public double DifficultyScore { get; set; } // 难度分数：(0.5*均值/5+0.3*标准差/2+0.2*最大值/5)*500
            public int FinalDifficulty { get; set; } // 最终难度：1-5
            public int EarlyPressureIndicator { get; set; } // 前期压力指标：从DockAfterTrioMatch第一个开始连续0的数量×3（范围0-21）
            public int TotalEarlyZeroCount { get; set; } // DockAfterTrioMatch前7个中0的总数×3（范围0-21）
            public int MaxConsecutiveZeroCount { get; set; } // DockAfterTrioMatch前7个中最长连续0的数量×3（范围0-21）
            public int ConsecutiveLowPressureCount { get; set; } // PressureValues从第一个开始连续1的数量
            public int TotalEarlyLowPressureCount { get; set; } // PressureValues前7个中1的总数

            public string ErrorMessage { get; set; }
        }

        /// <summary>
        /// 地形级筛选配置
        /// </summary>
        public class TerrainFilterConfig
        {
            public int TerrainId { get; set; }
            public (float min, float max)? PositionRange { get; set; }
            public (float min, float max)? ScoreRange { get; set; }
            public (int min, int max)? ConsecutiveLowPressureRange { get; set; }
            public (int min, int max)? TotalEarlyLowPressureRange { get; set; }
            public bool HasValidConfig => PositionRange.HasValue || ScoreRange.HasValue ||
                                         ConsecutiveLowPressureRange.HasValue || TotalEarlyLowPressureRange.HasValue;

            /// <summary>
            /// 检查分析结果是否符合地形特定筛选条件
            /// </summary>
            public bool MatchesCriteria(AnalysisResult result)
            {
                bool positionMatch = true;
                bool scoreMatch = true;
                bool consecutiveLowPressureMatch = true;
                bool totalEarlyLowPressureMatch = true;

                if (PositionRange.HasValue)
                {
                    var range = PositionRange.Value;
                    positionMatch = result.DifficultyPosition >= range.min && result.DifficultyPosition <= range.max;
                }

                if (ScoreRange.HasValue)
                {
                    var range = ScoreRange.Value;
                    scoreMatch = result.DifficultyScore >= range.min && result.DifficultyScore <= range.max;
                }

                if (ConsecutiveLowPressureRange.HasValue)
                {
                    var range = ConsecutiveLowPressureRange.Value;
                    consecutiveLowPressureMatch = result.ConsecutiveLowPressureCount >= range.min &&
                                                  result.ConsecutiveLowPressureCount <= range.max;
                }

                if (TotalEarlyLowPressureRange.HasValue)
                {
                    var range = TotalEarlyLowPressureRange.Value;
                    totalEarlyLowPressureMatch = result.TotalEarlyLowPressureCount >= range.min &&
                                                result.TotalEarlyLowPressureCount <= range.max;
                }

                return positionMatch && scoreMatch && consecutiveLowPressureMatch && totalEarlyLowPressureMatch;
            }

            /// <summary>
            /// 获取配置描述（简洁格式）
            /// </summary>
            public string GetDescription()
            {
                var parts = new List<string>();
                if (PositionRange.HasValue)
                    parts.Add($"Pos[{PositionRange.Value.min:F2}~{PositionRange.Value.max:F2}]");
                if (ScoreRange.HasValue)
                    parts.Add($"Score[{ScoreRange.Value.min:F0}~{ScoreRange.Value.max:F0}]");
                if (ConsecutiveLowPressureRange.HasValue)
                    parts.Add($"ConLP[{ConsecutiveLowPressureRange.Value.min}~{ConsecutiveLowPressureRange.Value.max}]");
                if (TotalEarlyLowPressureRange.HasValue)
                    parts.Add($"TotLP[{TotalEarlyLowPressureRange.Value.min}~{TotalEarlyLowPressureRange.Value.max}]");
                return parts.Count > 0 ? string.Join(" | ", parts) : "无筛选条件";
            }
        }

        /// <summary>
        /// 配置聚合键 - 用于分组同一配置的多个种子结果
        /// </summary>
        public class ConfigKey : IEquatable<ConfigKey>
        {
            public int TerrainId { get; set; }
            public string ExperienceModeStr { get; set; } // "[1,2,3]"格式
            public int ColorCount { get; set; }

            public override bool Equals(object obj)
            {
                return Equals(obj as ConfigKey);
            }

            public bool Equals(ConfigKey other)
            {
                return other != null &&
                       TerrainId == other.TerrainId &&
                       ExperienceModeStr == other.ExperienceModeStr &&
                       ColorCount == other.ColorCount;
            }

            public override int GetHashCode()
            {
                unchecked
                {
                    int hash = 17;
                    hash = hash * 31 + TerrainId.GetHashCode();
                    hash = hash * 31 + (ExperienceModeStr?.GetHashCode() ?? 0);
                    hash = hash * 31 + ColorCount.GetHashCode();
                    return hash;
                }
            }
        }

        /// <summary>
        /// 配置聚合结果 - 存储同一配置所有种子的平均值
        /// </summary>
        public class AggregatedResult
        {
            // 配置标识
            public int TerrainId { get; set; }
            public string LevelName { get; set; }
            public int[] ExperienceMode { get; set; }
            public int ColorCount { get; set; }
            public int TotalTiles { get; set; }
            public string AlgorithmName { get; set; }

            // 种子统计
            public int SeedCount { get; set; } // 运行的种子数量（包含成功和失败）
            public double WinRate { get; set; } // 胜率：成功通关的种子占总种子的百分比 (0.0-1.0)

            // 压力分析均值（11个核心字段，基于成功通关的种子计算）
            public double AvgTotalMoves { get; set; }
            public double AvgSuccessfulGroups { get; set; }
            public double AvgPeakDockCount { get; set; }
            public double AvgInitialMinCost { get; set; }
            public double AvgPressureValueMean { get; set; }
            public double AvgPressureValueMin { get; set; }
            public double AvgPressureValueMax { get; set; }
            public double AvgPressureValueStdDev { get; set; }
            public double AvgDifficultyScore { get; set; }
            public double AvgFinalDifficulty { get; set; }
            public double AvgEarlyPressureIndicator { get; set; }
            public double AvgTotalEarlyZeroCount { get; set; }
            public double AvgMaxConsecutiveZeroCount { get; set; }
            public double AvgConsecutiveLowPressureCount { get; set; }
            public double AvgTotalEarlyLowPressureCount { get; set; }
            public double AvgDifficultyPosition { get; set; }
        }

        /// <summary>
        /// 批量运行配置 - 简化版本，支持地形特定筛选
        /// </summary>
        [System.Serializable]
        public class RunConfig
        {
            [Header("=== CSV配置选择器 ===")]
            public int ExperienceConfigEnum = -2; // 体验模式枚举：1=exp-fix-1, 2=exp-fix-2, -1=exp-range-1所有配置, -2=数组排列组合
            public int ColorCountConfigEnum = -2; // 花色数量枚举：1=type-count-1, 2=type-count-2, -1=type-range-1所有配置

            [Header("=== 测试参数 ===")]
            public int TestLevelCount = 5; // 测试地形数量

            [Header("=== 排列组合配置 (ExperienceConfigEnum = -2时生效) ===")]
            public int ArrayLength = 3; // 数组长度
            public int MinValue = 1; // 最小值
            public int MaxValue = 9; // 最大值

            [Header("=== 配置选择策略 ===")]
            public bool UseRandomConfigSelection = false; // 是否随机选择配置：true=在范围内随机选择（不重复），false=按顺序遍历

            [Header("=== 随机种子配置 ===")]
            public bool UseFixedSeed = false; // 是否使用固定种子：true=结果可重现，false=完全随机
            public int[] FixedSeedValues = { 12345678, 11111111, 22222222, 33333333, 44444444, 55555555, 66666666, 77777777, 88888888, 99999999 }; // 固定种子值列表
            public int MaxSeedAttemptsPerConfig = 1000; // 每个配置最大种子尝试次数：在筛选模式下用于搜索符合条件的种子
            public int MaxEmptySeedAttemptsPerConfig = 100; // 每配置最大空运行次数：当配置连续x次都找不到符合条件的种子时提前退出

            [Header("=== 输出配置 ===")]
            public string OutputDirectory = "BattleAnalysisResults";
            public bool OutputPerConfigAverage = false; // 是否仅输出每配置平均值（同地形、同体验模式、同花色数量的所有种子的平均值）

            [Header("=== 筛选配置 ===")]
            public bool UseTerrainSpecificFiltering = false; // 是否使用地形特定筛选（从CSV读取）
            public bool EnableGlobalFiltering = true; // 是否启用全局筛选（作为fallback）
            public bool UseAverageFiltering = false; // 是否使用平均值筛选：true=跑满种子后对平均值筛选，false=每个种子立即筛选
            public float GlobalDifficultyPositionRangeMin = 0.55f; // 全局难点位置范围最小值
            public float GlobalDifficultyPositionRangeMax = 0.8f; // 全局难点位置范围最大值
            public float GlobalDifficultyScoreRangeMin = 150f; // 全局难度分数范围最小值
            public float GlobalDifficultyScoreRangeMax = 300f; // 全局难度分数范围最大值
            public int GlobalConsecutiveLowPressureRangeMin = 0; // 全局连续低压力范围最小值
            public int GlobalConsecutiveLowPressureRangeMax = 10; // 全局连续低压力范围最大值
            public int GlobalTotalEarlyLowPressureRangeMin = 0; // 全局前期低压力总数范围最小值
            public int GlobalTotalEarlyLowPressureRangeMax = 7; // 全局前期低压力总数范围最大值
            public int RequiredResultsPerTerrain = 1; // 每个地形需要找到的符合条件结果数量
            public int MaxConfigAttemptsPerTerrain = 1000; // 每个地形最大尝试配置数量

            /// <summary>
            /// 检查聚合结果是否符合筛选条件（用于平均值筛选模式）
            /// </summary>
            public bool MatchesCriteria(AggregatedResult result)
            {
                // 优先使用行特定筛选（如果启用）
                if (UseTerrainSpecificFiltering)
                {
                    // 注意：AggregatedResult没有RowIndex，无法直接使用TerrainFilterConfig
                    // 这里简化为使用全局筛选
                }

                // 使用全局筛选判断4个指标的平均值
                if (EnableGlobalFiltering)
                {
                    return result.AvgDifficultyPosition >= GlobalDifficultyPositionRangeMin &&
                           result.AvgDifficultyPosition <= GlobalDifficultyPositionRangeMax &&
                           result.AvgDifficultyScore >= GlobalDifficultyScoreRangeMin &&
                           result.AvgDifficultyScore <= GlobalDifficultyScoreRangeMax &&
                           result.AvgConsecutiveLowPressureCount >= GlobalConsecutiveLowPressureRangeMin &&
                           result.AvgConsecutiveLowPressureCount <= GlobalConsecutiveLowPressureRangeMax &&
                           result.AvgTotalEarlyLowPressureCount >= GlobalTotalEarlyLowPressureRangeMin &&
                           result.AvgTotalEarlyLowPressureCount <= GlobalTotalEarlyLowPressureRangeMax;
                }

                // 如果都没有启用筛选，返回true
                return true;
            }

            /// <summary>
            /// 获取用于测试的随机种子
            /// </summary>
            public int GetSeedForAttempt(int terrainId, int seedAttemptIndex)
            {
                if (UseFixedSeed)
                {
                    if (FixedSeedValues != null && FixedSeedValues.Length > 0)
                    {
                        int seedIndex = seedAttemptIndex % FixedSeedValues.Length;
                        return FixedSeedValues[seedIndex];
                    }
                    else
                    {
                        return 12345678 + terrainId * 10000 + seedAttemptIndex;
                    }
                }
                else
                {
                    return UnityEngine.Random.Range(1, int.MaxValue);
                }
            }

            /// <summary>
            /// 检查分析结果是否符合筛选条件（优先使用行特定配置）
            /// </summary>
            public bool MatchesCriteria(AnalysisResult result)
            {
                // 优先使用行特定筛选
                if (UseTerrainSpecificFiltering)
                {
                    var rowConfig = CsvConfigManager.GetTerrainFilterConfig(result.RowIndex);
                    if (rowConfig.HasValidConfig)
                    {
                        return rowConfig.MatchesCriteria(result);
                    }
                }

                // Fallback到全局筛选
                if (EnableGlobalFiltering)
                {
                    return result.DifficultyPosition >= GlobalDifficultyPositionRangeMin &&
                           result.DifficultyPosition <= GlobalDifficultyPositionRangeMax &&
                           result.DifficultyScore >= GlobalDifficultyScoreRangeMin &&
                           result.DifficultyScore <= GlobalDifficultyScoreRangeMax &&
                           result.ConsecutiveLowPressureCount >= GlobalConsecutiveLowPressureRangeMin &&
                           result.ConsecutiveLowPressureCount <= GlobalConsecutiveLowPressureRangeMax &&
                           result.TotalEarlyLowPressureCount >= GlobalTotalEarlyLowPressureRangeMin &&
                           result.TotalEarlyLowPressureCount <= GlobalTotalEarlyLowPressureRangeMax;
                }

                // 如果都没有启用筛选，返回true
                return true;
            }

            /// <summary>
            /// 检查是否启用了任何筛选
            /// </summary>
            public bool IsFilteringEnabled => UseTerrainSpecificFiltering || EnableGlobalFiltering;

            /// <summary>
            /// 获取筛选条件描述（简洁格式）
            /// </summary>
            public string GetFilterDescription()
            {
                if (!IsFilteringEnabled) return "筛选已禁用";

                var parts = new List<string>();
                if (UseTerrainSpecificFiltering)
                    parts.Add("地形特定筛选");
                if (EnableGlobalFiltering)
                {
                    var ranges = new List<string>
                    {
                        $"Pos[{GlobalDifficultyPositionRangeMin:F2}~{GlobalDifficultyPositionRangeMax:F2}]",
                        $"Score[{GlobalDifficultyScoreRangeMin:F0}~{GlobalDifficultyScoreRangeMax:F0}]",
                        $"ConLP[{GlobalConsecutiveLowPressureRangeMin}~{GlobalConsecutiveLowPressureRangeMax}]",
                        $"TotLP[{GlobalTotalEarlyLowPressureRangeMin}~{GlobalTotalEarlyLowPressureRangeMax}]"
                    };
                    parts.Add($"全局筛选: {string.Join(" | ", ranges)}");
                }

                return string.Join(" + ", parts) + $", 每地形需要{RequiredResultsPerTerrain}个结果";
            }

            /// <summary>
            /// 获取配置描述信息
            /// </summary>
            public string GetConfigDescription()
            {
                string expMode = ExperienceConfigEnum switch
                {
                    1 => "ExpFix1",
                    2 => "ExpFix2",
                    3 => "ExpFix3",
                    4 => "ExpFix4",
                    5 => "ExpFix5",
                    6 => "ExpFix6",
                    -1 => "所有ExpRange1配置",
                    -2 => $"排列组合[{MinValue}-{MaxValue}]^{ArrayLength}",
                    _ => $"配置{ExperienceConfigEnum}"
                };

                string colorMode = ColorCountConfigEnum switch
                {
                    1 => "TypeCount1",
                    2 => "TypeCount2",
                    3 => "TypeCount3",
                    4 => "TypeCount4",
                    5 => "TypeCount5",
                    6 => "TypeCount6",
                    -1 => "所有TypeRange1配置",
                    -2 => "动态花色范围(总tile数/3的40%-80%,上限25)",
                    _ => $"配置{ColorCountConfigEnum}"
                };

                string seedMode = UseFixedSeed ? $"固定种子列表({FixedSeedValues?.Length ?? 0}个)" : "随机种子";
                string filterMode = IsFilteringEnabled ? $", 筛选[{GetFilterDescription()}], 最多尝试{MaxConfigAttemptsPerTerrain}个配置, 每配置最多{MaxSeedAttemptsPerConfig}个种子" : "";

                return $"体验模式[{expMode}], 花色数量[{colorMode}], {seedMode}{filterMode}";
            }
        }

        private static List<CsvLevelConfig> _csvConfigs = null;
        private static readonly object _csvLock = new object();
        private static Dictionary<int, LevelData> _levelDataCache = new Dictionary<int, LevelData>();
        private static readonly Dictionary<int, List<int>> _standardColorsCache = new Dictionary<int, List<int>>();

        /// <summary>
        /// CSV配置管理器 - 统一的配置加载和解析服务
        /// </summary>
        public static class CsvConfigManager
        {
            /// <summary>
            /// 加载CSV配置数据 - 线程安全优化版本，按行存储
            /// </summary>
            public static void LoadCsvConfigs()
            {
                if (_csvConfigs != null) return;

                lock (_csvLock)
                {
                    if (_csvConfigs != null) return; // 双重检查锁定模式

                    _csvConfigs = new List<CsvLevelConfig>();

                    try
                    {
                        // 使用Path.Combine确保跨平台兼容
                        string csvPath = Path.Combine(Application.dataPath, "验证器", "Editor", "all_level.csv");

                        if (!File.Exists(csvPath))
                        {
                            Debug.LogError($"[BattleAnalyzer] CSV配置文件不存在: {csvPath}");
                            Debug.LogError($"[BattleAnalyzer] 请检查:");
                            Debug.LogError($"  1. 文件路径是否正确");
                            Debug.LogError($"  2. all_level.csv是否已提交到版本控制");
                            Debug.LogError($"  3. 检查.gitignore是否忽略了*.csv文件");
                            return;
                        }

                        Debug.Log($"[BattleAnalyzer] 开始加载CSV配置: {csvPath}");

                        int successCount = 0;
                        int failedCount = 0;

                        using (var fileStream = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.Read, FILE_BUFFER_SIZE))
                        using (var reader = new StreamReader(fileStream, Encoding.UTF8, true, FILE_BUFFER_SIZE)) // true=自动检测并移除BOM
                        {
                            string headerLine = reader.ReadLine(); // 跳过表头
                            Debug.Log($"[BattleAnalyzer] CSV表头: {headerLine?.Substring(0, Math.Min(100, headerLine?.Length ?? 0))}...");

                            string line;
                            int rowIndex = 0; // 行索引，从0开始

                            while ((line = reader.ReadLine()) != null)
                            {
                                var parts = CsvParser.ParseCsvLine(line);
                                if (parts.Length >= 17 && int.TryParse(parts[0], out int terrainId))
                                {
                                    var config = new CsvLevelConfig
                                    {
                                        RowIndex = rowIndex,
                                        TerrainId = terrainId,
                                        ExpFix1 = CsvParser.ParseIntArray(parts[1]),
                                        ExpFix2 = CsvParser.ParseIntArray(parts[2]),
                                        ExpFix3 = CsvParser.ParseIntArray(parts[3]),
                                        ExpFix4 = CsvParser.ParseIntArray(parts[4]),
                                        ExpFix5 = CsvParser.ParseIntArray(parts[5]),
                                        ExpFix6 = CsvParser.ParseIntArray(parts[6]),
                                        ExpRange1 = CsvParser.ParseIntArray(parts[7]),
                                        TypeCount1 = CsvParser.ParseIntOrDefault(parts[8], 1),
                                        TypeCount2 = CsvParser.ParseIntOrDefault(parts[9], 1),
                                        TypeCount3 = CsvParser.ParseIntOrDefault(parts[10], 1),
                                        TypeCount4 = CsvParser.ParseIntOrDefault(parts[11], 1),
                                        TypeCount5 = CsvParser.ParseIntOrDefault(parts[12], 1),
                                        TypeCount6 = CsvParser.ParseIntOrDefault(parts[13], 1),
                                        TypeRange1 = CsvParser.ParseIntOrDefault(parts[14], 1),
                                        PositionRange = CsvParser.ParseFloatRange(parts[15]),
                                        ScoreRange = CsvParser.ParseFloatRange(parts[16]),
                                        ConsecutiveLowPressureRange = CsvParser.ParseIntRange(parts[17]),
                                        TotalEarlyLowPressureRange = CsvParser.ParseIntRange(parts[18])
                                    };
                                    _csvConfigs.Add(config);
                                    successCount++;
                                    rowIndex++;
                                }
                                else
                                {
                                    failedCount++;
                                    Debug.LogWarning($"[BattleAnalyzer] CSV行{rowIndex + 2}解析失败:");
                                    Debug.LogWarning($"  列数: {parts.Length} (期望≥17)");
                                    Debug.LogWarning($"  第一列: [{parts[0]}] (TerrainId解析失败)");
                                    Debug.LogWarning($"  原始行: {line.Substring(0, Math.Min(100, line.Length))}...");

                                    // 检测是否是BOM导致的问题
                                    if (rowIndex == 0 && parts[0].Length > 0 && parts[0][0] > 127)
                                    {
                                        Debug.LogError($"[BattleAnalyzer] 警告: 第一行数据可能包含BOM字符!");
                                        Debug.LogError($"  第一列字节: {string.Join(" ", System.Text.Encoding.UTF8.GetBytes(parts[0]).Select(b => b.ToString("X2")))}");
                                        Debug.LogError($"  请检查CSV文件编码是否为UTF-8 without BOM");
                                    }
                                }
                            }
                        }

                        Debug.Log($"[BattleAnalyzer] CSV加载完成: 成功{successCount}行, 失败{failedCount}行, 总计{_csvConfigs.Count}行配置");
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[BattleAnalyzer] 加载CSV配置失败: {ex.GetType().Name} - {ex.Message}");
                        Debug.LogError($"[BattleAnalyzer] 堆栈跟踪: {ex.StackTrace}");
                        _csvConfigs = new List<CsvLevelConfig>();
                    }
                }
            }

            /// <summary>
            /// 根据行索引和枚举配置解析体验模式数组
            /// </summary>
            public static int[][] ResolveExperienceModesByRow(int experienceConfigEnum, int rowIndex, RunConfig runConfig = null)
            {
                LoadCsvConfigs();

                switch (experienceConfigEnum)
                {
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                        // 固定配置：使用特定行的配置
                        if (rowIndex < 0 || rowIndex >= _csvConfigs.Count)
                        {
                            Debug.LogWarning($"行索引 {rowIndex} 超出范围，使用默认值");
                            return new int[][] { new int[] { 1, 2, 3 } };
                        }

                        var config = _csvConfigs[rowIndex];
                        var selectedMode = experienceConfigEnum switch
                        {
                            1 => config.ExpFix1,
                            2 => config.ExpFix2,
                            3 => config.ExpFix3,
                            4 => config.ExpFix4,
                            5 => config.ExpFix5,
                            6 => config.ExpFix6,
                            _ => config.ExpFix1
                        };
                        return new int[][] { selectedMode };

                    case -1:
                        // 所有ExpRange1配置：返回全局去重后的所有配置
                        return GetAllExpRange1Configurations();

                    case -2:
                        // 排列组合配置
                        if (runConfig != null)
                        {
                            return GeneratePermutations(runConfig.ArrayLength, runConfig.MinValue, runConfig.MaxValue);
                        }
                        Debug.LogWarning("ExperienceConfigEnum = -2 需要提供 RunConfig 参数");
                        return new int[][] { new int[] { 1, 2, 3 } };

                    default:
                        Debug.LogWarning($"不支持的体验配置枚举: {experienceConfigEnum}，使用默认值");
                        return new int[][] { new int[] { 1, 2, 3 } };
                }
            }

            /// <summary>
            /// 根据枚举配置解析体验模式数组 - 支持-1全配置模式和-2排列组合模式（已废弃，请使用ResolveExperienceModesByRow）
            /// </summary>
            [Obsolete("请使用 ResolveExperienceModesByRow 方法")]
            public static int[][] ResolveExperienceModes(int experienceConfigEnum, int terrainId)
            {
                LoadCsvConfigs();

                switch (experienceConfigEnum)
                {
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                        // 固定配置：使用特定地形的第一行配置
                        var config = _csvConfigs.FirstOrDefault(c => c.TerrainId == terrainId);
                        if (config == null)
                        {
                            Debug.LogWarning($"未找到地形ID {terrainId} 的配置，使用默认值");
                            return new int[][] { new int[] { 1, 2, 3 } };
                        }

                        var selectedMode = experienceConfigEnum switch
                        {
                            1 => config.ExpFix1,
                            2 => config.ExpFix2,
                            3 => config.ExpFix3,
                            4 => config.ExpFix4,
                            5 => config.ExpFix5,
                            6 => config.ExpFix6,
                            _ => config.ExpFix1
                        };
                        return new int[][] { selectedMode };

                    case -1:
                        // 所有ExpRange1配置：返回全局去重后的所有配置
                        return GetAllExpRange1Configurations();

                    case -2:
                        // 排列组合配置：需要通过参数传递配置信息
                        Debug.LogWarning("ExperienceConfigEnum = -2 需要通过ResolveExperienceModesWithConfig方法使用");
                        return new int[][] { new int[] { 1, 2, 3 } };

                    default:
                        Debug.LogWarning($"不支持的体验配置枚举: {experienceConfigEnum}，使用默认值");
                        return new int[][] { new int[] { 1, 2, 3 } };
                }
            }

            /// <summary>
            /// 根据枚举配置和运行配置解析体验模式数组 - 支持-2排列组合模式
            /// </summary>
            public static int[][] ResolveExperienceModesWithConfig(int experienceConfigEnum, int terrainId, RunConfig runConfig)
            {
                if (experienceConfigEnum == -2)
                {
                    // 排列组合配置：生成从MinValue到MaxValue的所有长度为ArrayLength的排列组合
                    return GeneratePermutations(runConfig.ArrayLength, runConfig.MinValue, runConfig.MaxValue);
                }

                // 其他情况使用原有逻辑
                return ResolveExperienceModes(experienceConfigEnum, terrainId);
            }

            /// <summary>
            /// 生成排列组合：从minValue到maxValue的所有长度为arrayLength的组合
            /// 例如：从1,1,1,1到5,5,5,5
            /// </summary>
            private static int[][] GeneratePermutations(int arrayLength, int minValue, int maxValue)
            {
                // 计算总排列数：(maxValue - minValue + 1) ^ arrayLength
                int valueRange = maxValue - minValue + 1;
                int totalPermutations = (int)Math.Pow(valueRange, arrayLength);

                var permutations = new List<int[]>();

                // 生成所有排列组合
                for (int i = 0; i < totalPermutations; i++)
                {
                    var permutation = new int[arrayLength];
                    int temp = i;

                    // 将数字i转换为指定进制下的数组表示
                    for (int pos = 0; pos < arrayLength; pos++)
                    {
                        permutation[pos] = minValue + (temp % valueRange);
                        temp /= valueRange;
                    }

                    permutations.Add(permutation);
                }

                Debug.Log($"生成排列组合: 长度={arrayLength}, 范围=[{minValue}-{maxValue}], 总数={permutations.Count}");
                return permutations.ToArray();
            }

            /// <summary>
            /// 根据行索引和枚举配置解析花色数量数组
            /// </summary>
            public static int[] ResolveColorCountsByRow(int colorCountConfigEnum, int rowIndex)
            {
                LoadCsvConfigs();

                switch (colorCountConfigEnum)
                {
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                        // 固定配置：使用特定行的配置
                        if (rowIndex < 0 || rowIndex >= _csvConfigs.Count)
                        {
                            Debug.LogWarning($"行索引 {rowIndex} 超出范围，使用默认值");
                            return new int[] { 7 };
                        }

                        var config = _csvConfigs[rowIndex];
                        var selectedCount = colorCountConfigEnum switch
                        {
                            1 => config.TypeCount1,
                            2 => config.TypeCount2,
                            3 => config.TypeCount3,
                            4 => config.TypeCount4,
                            5 => config.TypeCount5,
                            6 => config.TypeCount6,
                            _ => config.TypeCount1
                        };
                        return new int[] { selectedCount };

                    case -1:
                        // 所有TypeRange1配置：返回全局去重后的所有配置
                        return GetAllTypeRange1Configurations();

                    case -2:
                        // 动态范围配置：基于总tile数/3的40%-80%范围遍历
                        if (rowIndex < 0 || rowIndex >= _csvConfigs.Count)
                        {
                            Debug.LogWarning($"行索引 {rowIndex} 超出范围，使用默认值");
                            return new int[] { 7 };
                        }
                        return GenerateDynamicColorRange(_csvConfigs[rowIndex].TerrainId);

                    default:
                        Debug.LogWarning($"不支持的花色配置枚举: {colorCountConfigEnum}，使用默认值");
                        return new int[] { 7 };
                }
            }

            /// <summary>
            /// 根据枚举配置解析花色数量数组 - 支持-1全配置模式，-2动态范围模式（已废弃，请使用ResolveColorCountsByRow）
            /// </summary>
            [Obsolete("请使用 ResolveColorCountsByRow 方法")]
            public static int[] ResolveColorCounts(int colorCountConfigEnum, int terrainId)
            {
                LoadCsvConfigs();

                switch (colorCountConfigEnum)
                {
                    case 1:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                        // 固定配置：使用特定地形的第一行配置
                        var config = _csvConfigs.FirstOrDefault(c => c.TerrainId == terrainId);
                        if (config == null)
                        {
                            Debug.LogWarning($"未找到地形ID {terrainId} 的配置，使用默认值");
                            return new int[] { 7 };
                        }

                        var selectedCount = colorCountConfigEnum switch
                        {
                            1 => config.TypeCount1,
                            2 => config.TypeCount2,
                            3 => config.TypeCount3,
                            4 => config.TypeCount4,
                            5 => config.TypeCount5,
                            6 => config.TypeCount6,
                            _ => config.TypeCount1
                        };
                        return new int[] { selectedCount };

                    case -1:
                        // 所有TypeRange1配置：返回全局去重后的所有配置
                        return GetAllTypeRange1Configurations();

                    case -2:
                        // 动态范围配置：基于总tile数/3的40%-80%范围遍历
                        return GenerateDynamicColorRange(terrainId);

                    default:
                        Debug.LogWarning($"不支持的花色配置枚举: {colorCountConfigEnum}，使用默认值");
                        return new int[] { 7 };
                }
            }

            private static int[][] _cachedExpRange1Configs;

            /// <summary>
            /// 获取exp-range-1列中所有不重复的体验配置 - 使用缓存优化
            /// </summary>
            private static int[][] GetAllExpRange1Configurations()
            {
                if (_cachedExpRange1Configs != null) return _cachedExpRange1Configs;

                LoadCsvConfigs();

                var uniqueConfigs = new HashSet<string>();
                var results = new List<int[]>();

                foreach (var row in _csvConfigs)
                {
                    string configStr = string.Join(",", row.ExpRange1);
                    if (uniqueConfigs.Add(configStr))
                    {
                        results.Add(row.ExpRange1);
                    }
                }

                _cachedExpRange1Configs = results.ToArray();
                Debug.Log($"获取到 {results.Count} 个不重复的ExpRange1配置");
                return _cachedExpRange1Configs;
            }

            private static int[] _cachedTypeRange1Configs;

            /// <summary>
            /// 生成基于总tile数的动态花色范围：总tile数/3的40%-80%，向下取整，上限25
            /// </summary>
            /// <param name="terrainId">地形ID</param>
            /// <returns>动态花色数量范围数组</returns>
            private static int[] GenerateDynamicColorRange(int terrainId)
            {
                try
                {
                    // 加载关卡数据获取总tile数
                    var levelData = LoadLevelData(terrainId.ToString());
                    if (levelData == null)
                    {
                        Debug.LogWarning($"无法加载关卡 {terrainId} 数据，使用默认范围");
                        return new int[] { 5, 6, 7, 8, 9, 10, 11, 12 }; // 默认范围
                    }

                    int totalTiles = CalculateTotalTileCount(levelData);
                    int totalGroups = totalTiles / 3; // 总组数

                    // 计算40%-80%范围，向下取整
                    int minColorCount = Mathf.FloorToInt(totalGroups * 0.4f);
                    int maxColorCount = Mathf.FloorToInt(totalGroups * 0.8f);

                    // 保证最小值至少为1，上限不超过25
                    minColorCount = Math.Max(1, minColorCount);
                    maxColorCount = Math.Min(25, Math.Max(minColorCount, maxColorCount));

                    // 生成范围数组
                    var result = new List<int>();
                    for (int i = minColorCount; i <= maxColorCount; i++)
                    {
                        result.Add(i);
                    }

                    Debug.Log($"地形{terrainId}动态花色范围: 总tiles={totalTiles}, 总组数={totalGroups}, " +
                             $"花色范围=[{minColorCount}-{maxColorCount}] ({result.Count}个配置)");

                    return result.ToArray();
                }
                catch (Exception ex)
                {
                    Debug.LogError($"生成动态花色范围失败 (地形{terrainId}): {ex.Message}");
                    return new int[] { 7 }; // 失败时返回默认值
                }
            }

            /// <summary>
            /// 获取type-range-1列中所有不重复的花色数量 - 使用缓存优化
            /// </summary>
            private static int[] GetAllTypeRange1Configurations()
            {
                if (_cachedTypeRange1Configs != null) return _cachedTypeRange1Configs;

                LoadCsvConfigs();

                var results = _csvConfigs
                    .Select(row => row.TypeRange1)
                    .Where(count => count > 0)
                    .Distinct()
                    .OrderBy(x => x)
                    .ToArray();

                _cachedTypeRange1Configs = results;
                Debug.Log($"获取到 {results.Length} 个不重复的TypeRange1配置: [{string.Join(",", results)}]");
                return _cachedTypeRange1Configs;
            }

            /// <summary>
            /// 兼容方法：获取单个体验模式（仅用于向后兼容）
            /// </summary>
            public static int[] ResolveExperienceMode(int experienceConfigEnum, int terrainId)
            {
                var modes = ResolveExperienceModes(experienceConfigEnum, terrainId);
                return modes.Length > 0 ? modes[0] : new int[] { 1, 2, 3 };
            }

            /// <summary>
            /// 兼容方法：获取单个花色数量（仅用于向后兼容）
            /// </summary>
            public static int ResolveColorCount(int colorCountConfigEnum, int terrainId)
            {
                var counts = ResolveColorCounts(colorCountConfigEnum, terrainId);
                return counts.Length > 0 ? counts[0] : 7;
            }

            /// <summary>
            /// 根据行索引获取筛选配置
            /// </summary>
            public static TerrainFilterConfig GetTerrainFilterConfig(int rowIndex)
            {
                LoadCsvConfigs();

                if (rowIndex >= 0 && rowIndex < _csvConfigs.Count)
                {
                    var config = _csvConfigs[rowIndex];
                    return new TerrainFilterConfig
                    {
                        TerrainId = config.TerrainId,
                        PositionRange = config.PositionRange,
                        ScoreRange = config.ScoreRange,
                        ConsecutiveLowPressureRange = config.ConsecutiveLowPressureRange,
                        TotalEarlyLowPressureRange = config.TotalEarlyLowPressureRange
                    };
                }

                return new TerrainFilterConfig { TerrainId = -1 };
            }

            /// <summary>
            /// 根据地形ID获取筛选配置（已废弃，使用第一个匹配的行）
            /// </summary>
            [Obsolete("请使用基于行索引的 GetTerrainFilterConfig(int rowIndex) 方法")]
            public static TerrainFilterConfig GetTerrainFilterConfigByTerrainId(int terrainId)
            {
                LoadCsvConfigs();

                var config = _csvConfigs.FirstOrDefault(c => c.TerrainId == terrainId);
                if (config != null)
                {
                    return new TerrainFilterConfig
                    {
                        TerrainId = terrainId,
                        PositionRange = config.PositionRange,
                        ScoreRange = config.ScoreRange,
                        ConsecutiveLowPressureRange = config.ConsecutiveLowPressureRange,
                        TotalEarlyLowPressureRange = config.TotalEarlyLowPressureRange
                    };
                }

                return new TerrainFilterConfig { TerrainId = terrainId };
            }
        }

        /// <summary>
        /// CSV解析工具类 - 提取通用解析逻辑，优化性能
        /// </summary>
        public static class CsvParser
        {
            private static readonly StringBuilder _reusableStringBuilder = new StringBuilder(CSV_PARSER_BUFFER_SIZE); // 复用StringBuilder

            /// <summary>
            /// 解析CSV行，处理引号包围的字段 - 优化内存版本
            /// </summary>
            public static string[] ParseCsvLine(string line)
            {
                var result = new List<string>();
                _reusableStringBuilder.Clear();
                bool inQuotes = false;

                for (int i = 0; i < line.Length; i++)
                {
                    char c = line[i];

                    if (c == '"')
                    {
                        inQuotes = !inQuotes;
                    }
                    else if (c == ',' && !inQuotes)
                    {
                        result.Add(_reusableStringBuilder.ToString());
                        _reusableStringBuilder.Clear();
                    }
                    else
                    {
                        _reusableStringBuilder.Append(c);
                    }
                }

                result.Add(_reusableStringBuilder.ToString());
                return result.ToArray();
            }

            /// <summary>
            /// 解析整数数组字符串，支持多种格式 - 保持原始逻辑确保一致性
            /// </summary>
            public static int[] ParseIntArray(string arrayStr)
            {
                try
                {
                    if (string.IsNullOrEmpty(arrayStr))
                        return new int[] { 1, 2, 3 };

                    arrayStr = arrayStr.Trim().Trim('[', ']', '(', ')', '{', '}');

                    if (string.IsNullOrEmpty(arrayStr))
                        return new int[] { 1, 2, 3 };

                    if (int.TryParse(arrayStr, out int singleValue))
                    {
                        return new int[] { singleValue, singleValue, singleValue };
                    }

                    var parts = arrayStr.Split(new char[] { ',', ' ', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    var result = new List<int>();

                    foreach (var part in parts)
                    {
                        if (int.TryParse(part.Trim(), out int value))
                        {
                            result.Add(value);
                        }
                    }

                    if (result.Count == 0)
                        return new int[] { 1, 2, 3 };
                    else if (result.Count == 1)
                        return new int[] { result[0], result[0], result[0] };
                    else if (result.Count == 2)
                        return new int[] { result[0], result[1], result[1] };
                    else
                        return result.ToArray(); // 支持任意长度，不再强制截断为3个元素
                }
                catch
                {
                    return new int[] { 1, 2, 3 };
                }
            }

            /// <summary>
            /// 解析整数或返回默认值 - 优化版本
            /// </summary>
            public static int ParseIntOrDefault(string str, int defaultValue)
            {
                return string.IsNullOrEmpty(str) || !int.TryParse(str.Trim(), out int result) ? defaultValue : result;
            }

            /// <summary>
            /// 解析范围值字符串为(min, max)元组
            /// 支持格式: "0.55-0.8", "0.55~0.8", "0.55,0.8", "150-300"
            /// </summary>
            public static (float min, float max)? ParseFloatRange(string rangeStr)
            {
                if (string.IsNullOrEmpty(rangeStr))
                    return null;

                rangeStr = rangeStr.Trim();
                if (string.IsNullOrEmpty(rangeStr))
                    return null;

                // 支持多种分隔符
                string[] separators = { "-", "~", ",", "，" };
                string[] parts = null;

                foreach (var separator in separators)
                {
                    if (rangeStr.Contains(separator))
                    {
                        parts = rangeStr.Split(new string[] { separator }, StringSplitOptions.RemoveEmptyEntries);
                        break;
                    }
                }

                if (parts == null || parts.Length != 2)
                    return null;

                if (float.TryParse(parts[0].Trim(), out float min) &&
                    float.TryParse(parts[1].Trim(), out float max))
                {
                    if (min <= max)
                        return (min, max);
                }

                return null;
            }

            /// <summary>
            /// 解析整数范围字符串（例如"0-10"或"0~10"）
            /// </summary>
            public static (int min, int max)? ParseIntRange(string rangeStr)
            {
                if (string.IsNullOrEmpty(rangeStr))
                    return null;

                rangeStr = rangeStr.Trim();
                if (string.IsNullOrEmpty(rangeStr))
                    return null;

                // 支持多种分隔符
                string[] separators = { "-", "~", ",", "，" };
                string[] parts = null;

                foreach (var separator in separators)
                {
                    if (rangeStr.Contains(separator))
                    {
                        parts = rangeStr.Split(new string[] { separator }, StringSplitOptions.RemoveEmptyEntries);
                        break;
                    }
                }

                if (parts == null || parts.Length != 2)
                    return null;

                if (int.TryParse(parts[0].Trim(), out int min) &&
                    int.TryParse(parts[1].Trim(), out int max))
                {
                    if (min <= max)
                        return (min, max);
                }

                return null;
            }
        }

        /// <summary>
        /// 游戏内单关卡压力分析 - 支持生成校验和筛选两种独立模式
        /// </summary>
        /// <param name="terrainId">地形ID (1-200)</param>
        /// <param name="experienceMode">体验配置数组，如[1,2,3]（筛选模式时被忽略）</param>
        /// <param name="colorCount">花色数量（筛选模式时被忽略）</param>
        /// <param name="randomSeed">随机种子（筛选模式时被忽略）</param>
        /// <param name="runCount">运行次数，默认1次（筛选模式时表示需要找到的结果数量）</param>
        /// <param name="positionRange">难点位置范围，如(0.55f, 0.8f)，null表示生成校验模式</param>
        /// <param name="scoreRange">难度分数范围，如(150f, 300f)，null表示生成校验模式</param>
        public static void AnalyzeSingleLevelPressure(int terrainId, int[] experienceMode, int colorCount, int randomSeed, int runCount = 1,
            (float min, float max)? positionRange = null, (float min, float max)? scoreRange = null)
        {
            // 判断运行模式
            bool isFilterMode = positionRange.HasValue || scoreRange.HasValue;

            if (isFilterMode)
            {
                // 筛选模式：在配置范围内搜索指定数量的符合条件结果
                RunFilterMode(terrainId, positionRange, scoreRange, runCount);
            }
            else
            {
                // 生成校验模式：使用指定参数生成并输出详细报告
                RunGenerationMode(terrainId, experienceMode, colorCount, randomSeed, runCount);
            }
        }

        /// <summary>
        /// 生成校验模式：根据指定参数生成关卡并输出详细分析报告
        /// </summary>
        private static void RunGenerationMode(int terrainId, int[] experienceMode, int colorCount, int randomSeed, int runCount)
        {
            Debug.Log($"=== 生成校验模式 ===");
            Debug.Log($"地形ID: {terrainId}, 体验配置: [{string.Join(",", experienceMode)}], 花色数量: {colorCount}");

            if (runCount == 1)
            {
                Debug.Log($"单次运行，随机种子: {randomSeed}");

                var result = RunSingleLevelAnalysis(terrainId.ToString(), experienceMode, colorCount, randomSeed);

                if (!string.IsNullOrEmpty(result.ErrorMessage))
                {
                    Debug.LogError($"分析失败: {result.ErrorMessage}");
                    return;
                }

                // 输出完整的压力分析结果
                Debug.Log($"=== 压力分析结果 ===");
                Debug.Log($"游戏状态: {(result.GameCompleted ? "通关成功" : "未通关")} ({result.CompletionStatus})");
                Debug.Log($"总步数: {result.TotalMoves}, 成功消除组数: {result.SuccessfulGroups}");
                Debug.Log($"峰值Dock数量: {result.PeakDockCount}, 开局最小Cost: {result.InitialMinCost}");
                Debug.Log($"压力值序列: [{string.Join(", ", result.PressureValues)}]");
                Debug.Log($"压力统计 - 均值: {result.PressureValueMean:F2}, 最小值: {result.PressureValueMin}, 最大值: {result.PressureValueMax}");
                Debug.Log($"压力标准差: {result.PressureValueStdDev:F2}, 难度分数: {result.DifficultyScore:F2}, 最终难度: {result.FinalDifficulty}/5");
                Debug.Log($"难点位置: {result.DifficultyPosition:F2} (0=开局, 1=结尾)");
                Debug.Log($"连续低压力数: {result.ConsecutiveLowPressureCount}, 前期低压力总数: {result.TotalEarlyLowPressureCount}");
            }
            else
            {
                Debug.Log($"多次运行模式: {runCount}次，使用随机种子");

                var validResults = new List<AnalysisResult>();
                var usedSeeds = new List<int>();

                // 进行多次运行
                for (int i = 0; i < runCount; i++)
                {
                    int currentSeed = UnityEngine.Random.Range(1, int.MaxValue);
                    usedSeeds.Add(currentSeed);

                    var result = RunSingleLevelAnalysis(terrainId.ToString(), experienceMode, colorCount, currentSeed);

                    if (string.IsNullOrEmpty(result.ErrorMessage))
                    {
                        validResults.Add(result);
                    }
                    else
                    {
                        Debug.LogWarning($"第{i+1}次运行失败(种子{currentSeed}): {result.ErrorMessage}");
                    }
                }

                if (validResults.Count == 0)
                {
                    Debug.LogError($"所有{runCount}次运行均失败！");
                    return;
                }

                // 输出多次运行统计结果
                var completedCount = validResults.Count(r => r.GameCompleted);
                var successfulResults = validResults.Where(r => r.GameCompleted).ToList();

                if (successfulResults.Count == 0)
                {
                    Debug.LogWarning($"所有{validResults.Count}次有效运行均未成功通关，无法计算成功关卡的均值统计");
                    return;
                }

                var avgTotalMoves = successfulResults.Average(r => r.TotalMoves);
                var avgSuccessfulGroups = successfulResults.Average(r => r.SuccessfulGroups);
                var avgPeakDockCount = successfulResults.Average(r => r.PeakDockCount);
                var avgInitialMinCost = successfulResults.Average(r => r.InitialMinCost);
                var avgPressureValueMean = successfulResults.Average(r => r.PressureValueMean);
                var avgPressureValueMin = successfulResults.Average(r => r.PressureValueMin);
                var avgPressureValueMax = successfulResults.Average(r => r.PressureValueMax);
                var avgPressureValueStdDev = successfulResults.Average(r => r.PressureValueStdDev);
                var avgDifficultyScore = successfulResults.Average(r => r.DifficultyScore);
                var avgFinalDifficulty = successfulResults.Average(r => r.FinalDifficulty);
                var avgDifficultyPosition = successfulResults.Average(r => r.DifficultyPosition);
                var avgConsecutiveLowPressureCount = successfulResults.Average(r => r.ConsecutiveLowPressureCount);
                var avgTotalEarlyLowPressureCount = successfulResults.Average(r => r.TotalEarlyLowPressureCount);

                Debug.Log($"=== 压力分析结果(均值) ===");
                Debug.Log($"有效运行数: {validResults.Count}/{runCount}, 通关成功率: {(float)completedCount/validResults.Count:P1}");
                Debug.Log($"成功关卡统计基数: {successfulResults.Count}个成功通关的关卡");
                Debug.Log($"使用的随机种子: [{string.Join(",", usedSeeds)}]");
                Debug.Log($"总步数(均值): {avgTotalMoves:F1}, 成功消除组数(均值): {avgSuccessfulGroups:F1}");
                Debug.Log($"峰值Dock数量(均值): {avgPeakDockCount:F1}, 开局最小Cost(均值): {avgInitialMinCost:F1}");
                Debug.Log($"压力统计(均值) - 均值: {avgPressureValueMean:F2}, 最小值: {avgPressureValueMin:F1}, 最大值: {avgPressureValueMax:F1}");
                Debug.Log($"压力标准差(均值): {avgPressureValueStdDev:F2}, 难度分数(均值): {avgDifficultyScore:F2}, 最终难度(均值): {avgFinalDifficulty:F1}/5");
                Debug.Log($"难点位置(均值): {avgDifficultyPosition:F2} (0=开局, 1=结尾) - 仅基于成功通关关卡");
                Debug.Log($"连续低压力数(均值): {avgConsecutiveLowPressureCount:F1}, 前期低压力总数(均值): {avgTotalEarlyLowPressureCount:F1}");
            }

            Debug.Log($"=== 生成校验完成 ===");
        }

        /// <summary>
        /// 筛选模式：在配置范围内搜索指定数量的符合条件的关卡配置
        /// </summary>
        private static void RunFilterMode(int terrainId, (float min, float max)? positionRange, (float min, float max)? scoreRange, int targetResultCount)
        {
            Debug.Log($"=== 筛选模式 ===");
            Debug.Log($"地形ID: {terrainId}");

            // 显示筛选条件
            var conditions = new List<string>();
            if (positionRange.HasValue)
                conditions.Add($"Position[{positionRange.Value.min:F2}-{positionRange.Value.max:F2}]");
            if (scoreRange.HasValue)
                conditions.Add($"Score[{scoreRange.Value.min:F0}-{scoreRange.Value.max:F0}]");
            Debug.Log($"筛选条件: {string.Join(", ", conditions)}");
            Debug.Log($"目标结果数量: {targetResultCount}个");

            // 创建临时RunConfig用于配置生成（模拟-2, -2配置）
            var tempConfig = new RunConfig
            {
                ArrayLength = 3,  // [a,b,c]格式
                MinValue = 1,     // 最小值
                MaxValue = 9      // 最大值
            };

            // 使用现有的配置生成逻辑
            var experienceModes = CsvConfigManager.ResolveExperienceModesWithConfig(-2, terrainId, tempConfig);
            var colorCounts = CsvConfigManager.ResolveColorCounts(-2, terrainId);

            var foundResults = new List<AnalysisResult>();
            int totalAttempts = 0;
            int maxAttempts = 10000; // 增加最大尝试次数

            Debug.Log($"开始搜索，配置空间: {experienceModes.Length}种体验模式 × {colorCounts.Length}种花色数量");

            // 在配置空间中搜索，找到目标数量后退出
            bool targetReached = false;
            foreach (var experienceMode in experienceModes)
            {
                if (targetReached) break;

                foreach (var colorCount in colorCounts)
                {
                    if (totalAttempts >= maxAttempts || foundResults.Count >= targetResultCount)
                    {
                        targetReached = true;
                        break;
                    }

                    totalAttempts++;
                    int randomSeed = UnityEngine.Random.Range(1, int.MaxValue);

                    var result = RunSingleLevelAnalysis(terrainId.ToString(), experienceMode, colorCount, randomSeed);
                    result.TerrainId = terrainId;

                    if (!string.IsNullOrEmpty(result.ErrorMessage)) continue;

                    // 检查是否符合筛选条件
                    if (CheckCriteria(result, positionRange, scoreRange))
                    {
                        foundResults.Add(result);
                        // 输出符合条件的结果
                        Debug.Log($"✓ 找到符合条件的配置 #{foundResults.Count}/{targetResultCount}:");
                        Debug.Log($"  体验模式: [{string.Join(",", result.ExperienceMode)}], 花色数量: {result.ColorCount}, 种子: {result.RandomSeed}");
                        Debug.Log($"  难点位置: {result.DifficultyPosition:F3}, 难度分数: {result.DifficultyScore:F1}, 最终难度: {result.FinalDifficulty}/5");
                        Debug.Log($"  峰值Dock: {result.PeakDockCount}, 压力均值: {result.PressureValueMean:F2}, 游戏状态: {result.CompletionStatus}");

                        // 达到目标数量后退出
                        if (foundResults.Count >= targetResultCount)
                        {
                            targetReached = true;
                            break;
                        }
                    }
                }
            }

            Debug.Log($"=== 筛选完成 ===");

            if (foundResults.Count >= targetResultCount)
            {
                Debug.Log($"成功找到目标数量: {foundResults.Count}/{targetResultCount}个符合条件的配置");
            }
            else if (totalAttempts >= maxAttempts)
            {
                Debug.Log($"达到最大尝试次数限制: {totalAttempts}次，找到 {foundResults.Count}/{targetResultCount}个符合条件的配置");
            }
            else
            {
                Debug.Log($"遍历完所有配置空间，找到 {foundResults.Count}/{targetResultCount}个符合条件的配置");
            }

            Debug.Log($"总搜索尝试: {totalAttempts}次");
        }

        /// <summary>
        /// 检查分析结果是否符合指定的筛选条件
        /// </summary>
        private static bool CheckCriteria(AnalysisResult result, (float min, float max)? positionRange, (float min, float max)? scoreRange)
        {
            bool positionMatch = true;
            bool scoreMatch = true;

            if (positionRange.HasValue)
            {
                var range = positionRange.Value;
                positionMatch = result.DifficultyPosition >= range.min && result.DifficultyPosition <= range.max;
            }

            if (scoreRange.HasValue)
            {
                var range = scoreRange.Value;
                scoreMatch = result.DifficultyScore >= range.min && result.DifficultyScore <= range.max;
            }

            return positionMatch && scoreMatch;
        }

        /// <summary>
        /// 运行单个关卡分析
        /// </summary>
        public static AnalysisResult RunSingleLevelAnalysis(string levelName, int[] experienceMode, int colorCount, int randomSeed = -1)
        {
            // 如果没有指定种子，生成随机种子
            if (randomSeed == -1)
            {
                randomSeed = UnityEngine.Random.Range(1, int.MaxValue);
            }

            // 设置随机种子
            UnityEngine.Random.InitState(randomSeed);

            var result = new AnalysisResult
            {
                LevelName = levelName,
                ExperienceMode = experienceMode,
                ColorCount = colorCount,
                RandomSeed = randomSeed
            };

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            try
            {
                // 1. 加载关卡数据
                var levelData = LoadLevelData(levelName);
                if (levelData == null)
                {
                    result.ErrorMessage = $"无法加载关卡数据: {levelName}";
                    return result;
                }

                result.TotalTiles = CalculateTotalTileCount(levelData);

                // 2. 创建Tile列表并分配花色
                var tiles = CreateTileListFromLevelData(levelData);
                var availableColors = CreateAvailableColors(colorCount);

                // 使用RuleBasedAlgorithm进行花色分配（内部已自动执行虚拟游戏模拟）
                var algorithm = new DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo.RuleBasedAlgorithm();
                algorithm.InitializeRandomSeed(randomSeed); // 确保算法使用相同的随机种子
                algorithm.AssignTileTypes(tiles, experienceMode, availableColors);

                // 获取真实使用的算法名称
                result.AlgorithmName = algorithm.AlgorithmName;

                // 3. 直接从算法获取动态复杂度评估结果（无需再次运行虚拟游戏）
                var dynamicResult = algorithm.LastDynamicComplexity;

                if (dynamicResult == null)
                {
                    result.ErrorMessage = "算法未返回动态复杂度评估结果";
                    return result;
                }

                if (!string.IsNullOrEmpty(dynamicResult.ErrorMessage))
                {
                    result.ErrorMessage = dynamicResult.ErrorMessage;
                    return result;
                }

                // 4. 映射动态评估结果到AnalysisResult
                result.GameCompleted = dynamicResult.GameCompleted;
                result.TotalMoves = dynamicResult.TotalMoves;
                result.CompletionStatus = dynamicResult.CompletionStatus;
                result.SuccessfulGroups = dynamicResult.SuccessfulGroups;
                result.TileIdSequence = dynamicResult.TileIdSequence;
                result.DockCountPerMove = dynamicResult.DockCountPerMove;
                result.PeakDockCount = dynamicResult.PeakDockCount;
                result.MinMovesToComplete = dynamicResult.MinMovesToComplete;
                result.InitialMinCost = dynamicResult.InitialMinCost;
                result.DifficultyPosition = dynamicResult.DifficultyPosition;
                result.DockAfterTrioMatch = dynamicResult.DockAfterTrioMatch;
                result.SafeOptionCounts = dynamicResult.SafeOptionCounts;
                result.MinCostAfterTrioMatch = dynamicResult.MinCostAfterTrioMatch;
                result.MinCostOptionsAfterTrioMatch = dynamicResult.MinCostOptionsAfterTrioMatch;
                result.PressureValues = dynamicResult.PressureValues;
                result.PressureValueMean = dynamicResult.PressureValueMean;
                result.PressureValueMin = dynamicResult.PressureValueMin;
                result.PressureValueMax = dynamicResult.PressureValueMax;
                result.PressureValueStdDev = dynamicResult.PressureValueStdDev;
                result.DifficultyScore = dynamicResult.DifficultyScore;
                result.FinalDifficulty = dynamicResult.FinalDifficulty;
                result.EarlyPressureIndicator = dynamicResult.EarlyPressureIndicator;
                result.TotalEarlyZeroCount = dynamicResult.TotalEarlyZeroCount;
                result.MaxConsecutiveZeroCount = dynamicResult.MaxConsecutiveZeroCount;
                result.ConsecutiveLowPressureCount = dynamicResult.ConsecutiveLowPressureCount;
                result.TotalEarlyLowPressureCount = dynamicResult.TotalEarlyLowPressureCount;

                stopwatch.Stop();
                result.GameDurationMs = (int)stopwatch.ElapsedMilliseconds;

                return result;
            }
            catch (Exception ex)
            {
                stopwatch.Stop();
                result.ErrorMessage = ex.Message;
                result.GameDurationMs = (int)stopwatch.ElapsedMilliseconds;
                return result;
            }
        }

        // ========================================
        // 【虚拟游戏逻辑已迁移】
        // 原SimulateAutoPlay、VirtualBattleAnalyzer等虚拟游戏相关方法（约686行）
        // 已完整迁移至 VirtualGameSimulator.cs
        // 现在通过 algorithm.LastDynamicComplexity 直接获取评估结果
        // ========================================

        /// <summary>
        /// 加载关卡数据 - 增加缓存机制优化
        /// </summary>
        private static LevelData LoadLevelData(string levelName)
        {
            try
            {
                // 直接解析levelName为数字ID作为缓存键
                int levelId = 0;
                int.TryParse(levelName, out levelId);

                // 检查缓存
                if (levelId > 0 && _levelDataCache.TryGetValue(levelId, out LevelData cachedLevel))
                {
                    return cachedLevel;
                }

                // 直接使用levelName作为JSON文件名
                string jsonFileName = levelName.EndsWith(".json") ? levelName : $"{levelName}.json";

                string jsonPath = Path.Combine(Application.dataPath, "..", "Tools", "Config", "Json", "Levels", jsonFileName);
                jsonPath = Path.GetFullPath(jsonPath);

                if (!File.Exists(jsonPath))
                {
                    Debug.LogError($"关卡JSON文件不存在: {jsonPath}");
                    return null;
                }

                string jsonContent = File.ReadAllText(jsonPath);
                var levelData = JsonUtility.FromJson<LevelData>(jsonContent);

                // 缓存结果
                if (levelId > 0 && levelData != null)
                {
                    _levelDataCache[levelId] = levelData;
                }

                return levelData;
            }
            catch (Exception ex)
            {
                Debug.LogError($"加载关卡数据失败 {levelName}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// 从LevelData创建Tile列表
        /// </summary>
        private static List<Tile> CreateTileListFromLevelData(LevelData levelData)
        {
            var tiles = new List<Tile>();

            foreach (var layer in levelData.Layers)
            {
                foreach (var tileData in layer.tiles)
                {
                    var tile = new Tile(tileData);

                    if (tileData.IsConst)
                    {
                        tile.SetElementValue(tileData.ConstElementValue);
                    }

                    tiles.Add(tile);
                }
            }

            return tiles;
        }

        /// <summary>
        /// 从LevelDatabase获取可用的瓦片花色池
        /// </summary>
        private static List<int> GetAvailableColorsFromDatabase()
        {
            try
            {
#if UNITY_EDITOR
                // Editor模式：使用AssetDatabase加载
                var levelDatabase = UnityEditor.AssetDatabase.LoadAssetAtPath<LevelDatabase>(
                    "Assets/ArtRes/TMRes/StaticSettings/LevelDatabase.asset");
#else
                // 运行时模式：尝试从Resources加载（如果有的话）
                var levelDatabase = UnityEngine.Resources.Load<LevelDatabase>("LevelDatabase");
#endif

                if (levelDatabase == null)
                {
                    Debug.LogWarning("[BattleAnalyzer] 无法加载LevelDatabase，使用备用花色池");
                    return GetFallbackColorPool();
                }

                // 从LevelDatabase获取所有有效的ElementValue
                var allColors = levelDatabase.Tiles
                    .Where(tile => tile != null && tile.ElementValue > 0)
                    .Select(tile => tile.ElementValue)
                    .Distinct()
                    .OrderBy(x => x)
                    .ToList();

                if (allColors.Count == 0)
                {
                    Debug.LogWarning("[BattleAnalyzer] LevelDatabase中无有效瓦片，使用备用花色池");
                    return GetFallbackColorPool();
                }

                Debug.Log($"[BattleAnalyzer] 从LevelDatabase加载花色池成功，共{allColors.Count}种花色: [{string.Join(", ", allColors)}]");
                return allColors;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[BattleAnalyzer] 加载LevelDatabase失败: {ex.Message}，使用备用花色池");
                return GetFallbackColorPool();
            }
        }

        /// <summary>
        /// 备用花色池 - 仅在无法访问LevelDatabase时使用
        /// 包含9个色系的所有常见花色，总计约100种花色作为保险
        /// </summary>
        private static List<int> GetFallbackColorPool()
        {
            var fallbackColors = new List<int>();

            // 🔴 100系列 - 红色系 (预留100-199)
            for (int i = 101; i <= 120; i++) fallbackColors.Add(i);

            // 🟠 200系列 - 橙色系 (预留200-299)
            for (int i = 201; i <= 220; i++) fallbackColors.Add(i);

            // 🟡 300系列 - 黄色系 (预留300-399)
            for (int i = 301; i <= 320; i++) fallbackColors.Add(i);

            // 🟢 400系列 - 绿色系 (预留400-499)
            for (int i = 401; i <= 420; i++) fallbackColors.Add(i);

            // 🩵 500系列 - 青色系 (预留500-599)
            for (int i = 501; i <= 520; i++) fallbackColors.Add(i);

            // 🔵 600系列 - 蓝色系 (预留600-699)
            for (int i = 601; i <= 620; i++) fallbackColors.Add(i);

            // 🟣 700系列 - 紫色系 (预留700-799)
            for (int i = 701; i <= 720; i++) fallbackColors.Add(i);

            // ⚫ 800系列 - 黑色系 (预留800-899)
            for (int i = 801; i <= 820; i++) fallbackColors.Add(i);

            // ⚪ 900系列 - 白色系 (预留900-999)
            for (int i = 901; i <= 920; i++) fallbackColors.Add(i);

            Debug.LogWarning($"[BattleAnalyzer] 使用备用花色池，共{fallbackColors.Count}种预定义花色 (101-120, 201-220, ..., 901-920)");
            return fallbackColors;
        }

        /// <summary>
        /// 获取花色池大小 - 用于提前检查花色数量是否合法
        /// </summary>
        private static int GetColorPoolSize()
        {
            // 使用缓存的花色池计算大小
            if (_colorPoolCache != null)
            {
                return _colorPoolCache.Count;
            }

            // 首次调用时加载花色池
            _colorPoolCache = GetAvailableColorsFromDatabase();
            return _colorPoolCache.Count;
        }

        private static List<int> _colorPoolCache = null; // 花色池缓存

        /// <summary>
        /// 创建可用花色列表 - 基于LevelDatabase动态获取花色池
        /// </summary>
        private static List<int> CreateAvailableColors(int colorCount)
        {
            // 检查缓存
            if (_standardColorsCache.TryGetValue(colorCount, out List<int> cachedColors))
            {
                return new List<int>(cachedColors); // 返回副本防止修改
            }

            // 从缓存或LevelDatabase获取完整花色池
            if (_colorPoolCache == null)
            {
                _colorPoolCache = GetAvailableColorsFromDatabase();
            }
            var fullColorPool = _colorPoolCache;

            // 检查花色数量是否合法（这一步应该在调用前完成，这里仅作为防御性检查）
            if (colorCount > fullColorPool.Count)
            {
                Debug.LogError($"[BattleAnalyzer] 请求的花色数量({colorCount})超过花色池大小({fullColorPool.Count})，无法创建花色列表！");
                return new List<int>(); // 返回空列表
            }

            // 使用原始Fisher-Yates洗牌算法（正向遍历）保持结果一致性
            var shuffled = new List<int>(fullColorPool);
            for (int i = 0; i < shuffled.Count; i++)
            {
                int randomIndex = UnityEngine.Random.Range(i, shuffled.Count);
                (shuffled[i], shuffled[randomIndex]) = (shuffled[randomIndex], shuffled[i]);
            }
            var result = shuffled.GetRange(0, colorCount);

            // 缓存结果
            _standardColorsCache[colorCount] = new List<int>(result);
            return result;
        }

        /// <summary>
        /// 计算关卡总瓦片数量
        /// </summary>
        private static int CalculateTotalTileCount(LevelData levelData)
        {
            return levelData.Layers.Sum(layer => layer.tiles.Length);
        }

        /// <summary>
        /// 批量运行分析 - 支持筛选和策略性配置切换的增强版本
        /// </summary>
        public static List<AnalysisResult> RunBatchAnalysis(RunConfig config)
        {
            CsvConfigManager.LoadCsvConfigs();
            var results = new List<AnalysisResult>();

            // 获取花色池大小并输出信息
            int maxColorPoolSize = GetColorPoolSize();
            Debug.Log($"[BattleAnalyzer] 花色池大小: {maxColorPoolSize}种花色");

            // 应用种子配置
            if (config.UseFixedSeed)
            {
                var seedList = config.FixedSeedValues != null && config.FixedSeedValues.Length > 0
                    ? string.Join(",", config.FixedSeedValues)
                    : "默认种子";
                Debug.Log($"使用固定随机种子列表: [{seedList}] (确保结果可重现)");
            }
            else
            {
                Debug.Log("使用随机种子模式 (每次结果不同)");
            }

            Debug.Log($"开始批量分析: {config.GetConfigDescription()}");
            Debug.Log("[AlgoLogger] 编辑器模式下关卡生成内部日志已自动抑制");

            // 内存监控初始化 - 使用流式写入避免StringBuilder爆炸
            long initialMemory = GC.GetTotalMemory(false) / 1024 / 1024;
            int initialGen0 = GC.CollectionCount(0);
            int initialGen1 = GC.CollectionCount(1);

            // 创建内存监控日志文件并打开流式写入
            string memoryLogPath = Path.Combine(config.OutputDirectory, $"MemoryMonitor_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
            StreamWriter memoryLogWriter = null;
            try
            {
                var directory = Path.GetDirectoryName(memoryLogPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                memoryLogWriter = new StreamWriter(memoryLogPath, false, Encoding.UTF8);
                memoryLogWriter.AutoFlush = true; // 立即刷新,防止崩溃丢失数据
                memoryLogWriter.WriteLine("=== 内存监控日志 ===");
                memoryLogWriter.WriteLine($"开始时间,{DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                memoryLogWriter.WriteLine($"初始内存,{initialMemory}MB");
                memoryLogWriter.WriteLine("进度,已用内存(MB),内存增长(MB),GC Gen0次数,GC Gen1次数,时间戳");
                Debug.Log($"[内存监控] 初始内存: {initialMemory}MB, 日志文件: {memoryLogPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[内存监控] 无法创建日志文件: {ex.Message}");
            }

            // 预计算总任务数 - 按行配置处理
            int totalTasks = 0;
            var rowConfigs = new List<(int rowIndex, int terrainId, int[][] experienceModes, int[] colorCounts)>();

            // 检查CSV配置加载状态
            if (_csvConfigs == null || _csvConfigs.Count == 0)
            {
                Debug.LogError("CSV配置加载失败或为空，无法进行批量分析！");
                return results;
            }

            Debug.Log($"CSV配置加载成功，共 {_csvConfigs.Count} 行配置");

            // 按行索引顺序处理，取前 TestLevelCount 行
            int rowsToProcess = Math.Min(config.TestLevelCount, _csvConfigs.Count);
            for (int i = 0; i < rowsToProcess; i++)
            {
                var rowConfig = _csvConfigs[i];
                int rowIndex = rowConfig.RowIndex;
                int terrainId = rowConfig.TerrainId;

                // 解析当前行的配置
                var experienceModes = CsvConfigManager.ResolveExperienceModesByRow(config.ExperienceConfigEnum, rowIndex, config);
                var colorCounts = CsvConfigManager.ResolveColorCountsByRow(config.ColorCountConfigEnum, rowIndex);

                rowConfigs.Add((rowIndex, terrainId, experienceModes, colorCounts));

                // 计算任务数
                if (config.IsFilteringEnabled)
                {
                    totalTasks += config.RequiredResultsPerTerrain; // 期望找到的结果数量
                }
                else
                {
                    totalTasks += experienceModes.Length * colorCounts.Length * config.MaxSeedAttemptsPerConfig;
                }
            }

            Debug.Log($"处理行数: {rowConfigs.Count}, 总任务数: {totalTasks}");
            Debug.Log($"行索引列表: [{string.Join(",", rowConfigs.Select(r => $"Row{r.rowIndex}(T{r.terrainId})"))}]");

            // 预分配结果列表容量优化
            results.Capacity = totalTasks;
            int completedTasks = 0;
            int skippedTasks = 0;
            int uniqueIdCounter = 1; // 唯一ID计数器

            // 按行独立处理每个配置
            foreach (var (rowIndex, terrainId, experienceModes, colorCounts) in rowConfigs)
            {
                string levelName = terrainId.ToString();

                // 行状态追踪（替代原来的"地形状态"）
                int configAttempts = 0;
                bool rowCompleted = false; // 标记当前行是否已完成

                // 生成配置组合列表（体验模式 × 花色数量）
                var allConfigCombinations = new List<(int[] expMode, int colorCount)>();
                foreach (var expMode in experienceModes)
                {
                    foreach (var colorCount in colorCounts)
                    {
                        allConfigCombinations.Add((expMode, colorCount));
                    }
                }

                // 根据配置决定遍历顺序
                IEnumerable<(int[] expMode, int colorCount)> configSequence;
                if (config.UseRandomConfigSelection)
                {
                    // 随机打乱配置顺序（不重复）
                    // 使用terrainId和时间戳生成种子,确保每个地形的随机序列不同
                    var shuffledConfigs = new List<(int[] expMode, int colorCount)>(allConfigCombinations);
                    var random = new System.Random(terrainId * 1000 + DateTime.Now.Millisecond);

                    // Fisher-Yates洗牌算法
                    for (int i = shuffledConfigs.Count - 1; i > 0; i--)
                    {
                        int j = random.Next(i + 1);
                        var temp = shuffledConfigs[i];
                        shuffledConfigs[i] = shuffledConfigs[j];
                        shuffledConfigs[j] = temp;
                    }
                    configSequence = shuffledConfigs;
                    Debug.Log($"[行{rowIndex}|地形{terrainId}] 随机配置模式：共{shuffledConfigs.Count}个配置组合，已随机打乱顺序");
                }
                else
                {
                    // 按原顺序遍历
                    configSequence = allConfigCombinations;
                }

                // 遍历配置组合
                foreach (var (experienceMode, colorCount) in configSequence)
                {
                    if (rowCompleted) break;
                    if (configAttempts >= config.MaxConfigAttemptsPerTerrain) break;

                    // 检查花色数量是否超过花色池上限
                    if (colorCount > maxColorPoolSize)
                    {
                        Debug.Log($"跳过配置 (行{rowIndex}|地形{terrainId}): 花色数量({colorCount})超过花色池上限({maxColorPoolSize})");
                        continue;
                    }

                    configAttempts++;
                    var currentConfigResults = new List<AnalysisResult>(); // 当前配置找到的符合条件结果
                    int seedAttempts = 0; // 当前配置的种子尝试次数

                    // 平均值筛选模式：固定跑满所有种子
                    if (config.IsFilteringEnabled && config.UseAverageFiltering)
                    {
                        // 跑满MaxSeedAttemptsPerConfig个种子
                        for (int i = 0; i < config.MaxSeedAttemptsPerConfig; i++)
                        {
                            int randomSeed = config.GetSeedForAttempt(terrainId, i);
                            seedAttempts++;
                            completedTasks++;

#if VERBOSE_ANALYZER_LOGGING
                            Debug.Log($"[行{rowIndex}|地形{terrainId}] 配置#{configAttempts} 平均值筛选种子{seedAttempts}/{config.MaxSeedAttemptsPerConfig}: " +
                                     $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}");
#else
                            // 正常模式：每500个任务输出一次进度（降低日志频率防止Console爆炸）
                            if (completedTasks % 500 == 0 || completedTasks == 1)
                            {
                                Debug.Log($"[进度] {completedTasks}/{totalTasks} ({100f * completedTasks / totalTasks:F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                            }
#endif

                            // 运行单次分析
                            var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                            result.RowIndex = rowIndex;
                            result.TerrainId = terrainId;
                            result.UniqueId = $"BA_{uniqueIdCounter:D6}";
                            uniqueIdCounter++;

                            // 缓存所有结果
                            currentConfigResults.Add(result);
                        }

                        // 计算平均值并判断是否符合条件
                        var aggregatedResults = AggregateResultsByConfig(currentConfigResults);
                        if (aggregatedResults.Count > 0)
                        {
                            var aggregated = aggregatedResults[0];
                            if (config.MatchesCriteria(aggregated))
                            {
                                // 平均值符合条件，添加所有结果到总结果集
                                results.AddRange(currentConfigResults);
                                rowCompleted = true;
                                Debug.Log($"  ✓✓✓ 平均值符合条件！行{rowIndex}|地形{terrainId}配置[{string.Join(",", experienceMode)}]花色{colorCount} " +
                                         $"平均值: Pos={aggregated.AvgDifficultyPosition:F3}, Score={aggregated.AvgDifficultyScore:F1}, " +
                                         $"ConLP={aggregated.AvgConsecutiveLowPressureCount:F1}, TotLP={aggregated.AvgTotalEarlyLowPressureCount:F1}");
                            }
                            else
                            {
                                Debug.LogWarning($"  ✗ 平均值不符合条件 (行{rowIndex}|地形{terrainId}): 配置[{string.Join(",", experienceMode)}]花色{colorCount} " +
                                               $"平均值: Pos={aggregated.AvgDifficultyPosition:F3}, Score={aggregated.AvgDifficultyScore:F1}, " +
                                               $"ConLP={aggregated.AvgConsecutiveLowPressureCount:F1}, TotLP={aggregated.AvgTotalEarlyLowPressureCount:F1}");
                            }
                        }
                    }
                    else
                    {
                        // 即时筛选模式 或 非筛选模式
                        while (seedAttempts < config.MaxSeedAttemptsPerConfig)
                        {
                            // 提前退出检测：配置空运行上限
                            if (config.IsFilteringEnabled &&
                                seedAttempts >= config.MaxEmptySeedAttemptsPerConfig &&
                                currentConfigResults.Count == 0)
                            {
                                Debug.LogWarning($"配置空运行退出 (行{rowIndex}|地形{terrainId}): 配置[{string.Join(",", experienceMode)}]花色{colorCount}尝试{seedAttempts}个种子后未找到任何符合条件的结果，提前退出");
                                break; // 退出种子循环，进入下一个配置
                            }

                            int randomSeed = config.GetSeedForAttempt(terrainId, seedAttempts);
                            seedAttempts++;
                            completedTasks++;

#if VERBOSE_ANALYZER_LOGGING
                            // 详细模式：输出每个任务的详细信息
                            if (config.IsFilteringEnabled)
                            {
                                var terrainConfig = CsvConfigManager.GetTerrainFilterConfig(rowIndex);
                                string filterInfo = terrainConfig.HasValidConfig ? $"行筛选[{terrainConfig.GetDescription()}]" : "全局筛选";
                                Debug.Log($"[行{rowIndex}|地形{terrainId}] 配置#{configAttempts} 种子尝试{seedAttempts}/{config.MaxSeedAttemptsPerConfig}: " +
                                         $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}, {filterInfo} " +
                                         $"(当前配置已找到{currentConfigResults.Count}个, 配置总需求{config.RequiredResultsPerTerrain}个)");
                            }
                            else
                            {
                                Debug.Log($"[{completedTasks}/{totalTasks - skippedTasks}] 分析关卡 行{rowIndex}|地形{terrainId}: " +
                                         $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}");
                            }
#else
                            // 正常模式：每500个任务输出一次进度（降低日志频率防止Console爆炸）
                            if (completedTasks % 500 == 0 || completedTasks == 1)
                            {
                                Debug.Log($"[进度] {completedTasks}/{totalTasks - skippedTasks} ({100f * completedTasks / (totalTasks - skippedTasks):F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                            }
#endif

                            // 运行单次分析
                            var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                            result.RowIndex = rowIndex;
                            result.TerrainId = terrainId;
                            result.UniqueId = $"BA_{uniqueIdCounter:D6}";
                            uniqueIdCounter++;

                            // 内存监控：每100个任务检查一次
                            if (completedTasks % 100 == 0)
                            {
                                long currentMemory = GC.GetTotalMemory(false) / 1024 / 1024;
                                long memoryGrowth = currentMemory - initialMemory;
                                int gen0Count = GC.CollectionCount(0) - initialGen0;
                                int gen1Count = GC.CollectionCount(1) - initialGen1;
                                string timestamp = DateTime.Now.ToString("HH:mm:ss");

                                // 流式写入内存监控日志,避免内存累积
                                try
                                {
                                    memoryLogWriter?.WriteLine($"{completedTasks},{currentMemory},{memoryGrowth},{gen0Count},{gen1Count},{timestamp}");
                                }
                                catch { /* 忽略写入失败 */ }

                                // 内存警告阈值 - 降低阈值提前警告
                                if (memoryGrowth > 300)
                                {
                                    Debug.LogWarning($"[内存监控] ⚠️ 内存增长过高: {memoryGrowth}MB (当前{currentMemory}MB)");
                                }

                                // 增强GC频率：每200个任务或内存增长超过150MB时立即GC
                                if (completedTasks % 200 == 0 || memoryGrowth > 150)
                                {
                                    GC.Collect(1, GCCollectionMode.Forced); // 强制GC,包括Gen1
                                    GC.WaitForPendingFinalizers();
                                    long memoryAfterGC = GC.GetTotalMemory(true) / 1024 / 1024;
                                    if (completedTasks % 1000 == 0) // 降低GC日志频率
                                    {
                                        Debug.Log($"[内存监控] 执行GC: {currentMemory}MB → {memoryAfterGC}MB (回收{currentMemory - memoryAfterGC}MB)");
                                    }
                                }
                            }

                            // 筛选模式：检查是否符合条件
                            if (config.IsFilteringEnabled)
                            {
                                if (config.MatchesCriteria(result))
                                {
                                    currentConfigResults.Add(result);
                                    Debug.Log($"  ✓ 找到符合条件的种子！DifficultyPosition={result.DifficultyPosition:F3}, DifficultyScore={result.DifficultyScore:F1} " +
                                             $"(当前配置第{currentConfigResults.Count}个, 配置总需求{config.RequiredResultsPerTerrain}个)");

                                    // 检查当前配置是否已找到足够数量（单配置独立达标）
                                    if (currentConfigResults.Count >= config.RequiredResultsPerTerrain)
                                    {
                                        rowCompleted = true;
                                        Debug.Log($"  ✓✓✓ 配置达标！行{rowIndex}|地形{terrainId}配置[{string.Join(",", experienceMode)}]花色{colorCount}找到{currentConfigResults.Count}个符合条件的结果");
                                        break; // 退出种子循环
                                    }
                                }
                            }
                            else
                            {
                                // 非筛选模式：直接添加所有结果
                                currentConfigResults.Add(result);
                            }
                        }

                        // 检查行是否已完成或配置未达标
                        if (config.IsFilteringEnabled)
                        {
                            if (rowCompleted)
                            {
                                // 行达标：添加当前配置的结果并退出
                                results.AddRange(currentConfigResults);
                                Debug.Log($"行 {rowIndex}|地形{terrainId} 成功完成：配置[{string.Join(",", experienceMode)}]花色{colorCount}找到{currentConfigResults.Count}个符合条件的结果");
                                break; // 退出配置循环
                            }
                            else if (seedAttempts >= config.MaxSeedAttemptsPerConfig && currentConfigResults.Count > 0 && currentConfigResults.Count < config.RequiredResultsPerTerrain)
                            {
                                // 配置未达标：舍弃不完整的结果
                                Debug.LogWarning($"配置未达标舍弃 (行{rowIndex}|地形{terrainId}): 配置[{string.Join(",", experienceMode)}]花色{colorCount}尝试{seedAttempts}个种子后仅找到{currentConfigResults.Count}/{config.RequiredResultsPerTerrain}个符合条件的结果，舍弃这些结果");
                            }
                        }
                        else
                        {
                            // 非筛选模式：直接添加所有结果
                            results.AddRange(currentConfigResults);
                        }
                    }
                }
            }

            // 写入最终统计并关闭内存监控日志流
            long finalMemory = GC.GetTotalMemory(false) / 1024 / 1024;
            long totalMemoryGrowth = finalMemory - initialMemory;
            try
            {
                if (memoryLogWriter != null)
                {
                    memoryLogWriter.WriteLine($"=== 最终统计 ===");
                    memoryLogWriter.WriteLine($"结束时间,{DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    memoryLogWriter.WriteLine($"最终内存,{finalMemory}MB");
                    memoryLogWriter.WriteLine($"总内存增长,{totalMemoryGrowth}MB");
                    memoryLogWriter.WriteLine($"总GC次数,Gen0={GC.CollectionCount(0) - initialGen0}, Gen1={GC.CollectionCount(1) - initialGen1}");
                    memoryLogWriter.WriteLine($"总任务数,{completedTasks}");
                    memoryLogWriter.WriteLine($"平均每任务内存,{(completedTasks > 0 ? totalMemoryGrowth * 1024.0 / completedTasks : 0):F2}KB");
                    memoryLogWriter.Close();
                    memoryLogWriter.Dispose();
                    Debug.Log($"[内存监控] 日志已保存: {memoryLogPath}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[内存监控] 保存日志失败: {ex.Message}");
            }
            finally
            {
                // 确保资源释放
                memoryLogWriter?.Dispose();
            }

            Debug.Log($"[内存监控] 最终内存: {finalMemory}MB, 总增长: {totalMemoryGrowth}MB");

            if (config.IsFilteringEnabled)
            {
                Debug.Log($"筛选分析完成: 找到 {results.Count} 个符合条件的结果, 跳过 {skippedTasks} 个超限任务");
            }
            else
            {
                Debug.Log($"批量分析完成: {results.Count} 个任务结果, 跳过 {skippedTasks} 个超限任务");
            }
            return results;
        }

        /// <summary>
        /// 将结果按配置聚合，计算每个配置的平均值
        /// </summary>
        public static List<AggregatedResult> AggregateResultsByConfig(List<AnalysisResult> results)
        {
            // 按ConfigKey分组
            var groupedResults = new Dictionary<ConfigKey, List<AnalysisResult>>();

            foreach (var result in results)
            {
                var key = new ConfigKey
                {
                    TerrainId = result.TerrainId,
                    ExperienceModeStr = $"[{string.Join(",", result.ExperienceMode)}]",
                    ColorCount = result.ColorCount
                };

                if (!groupedResults.ContainsKey(key))
                    groupedResults[key] = new List<AnalysisResult>();

                groupedResults[key].Add(result);
            }

            // 对每组计算平均值
            var aggregatedResults = new List<AggregatedResult>();

            foreach (var kvp in groupedResults)
            {
                var configResults = kvp.Value;
                var successfulResults = configResults.Where(r => r.GameCompleted).ToList();
                var firstResult = configResults.First();

                var aggregated = new AggregatedResult
                {
                    TerrainId = firstResult.TerrainId,
                    LevelName = firstResult.LevelName,
                    ExperienceMode = firstResult.ExperienceMode,
                    ColorCount = firstResult.ColorCount,
                    TotalTiles = firstResult.TotalTiles,
                    AlgorithmName = firstResult.AlgorithmName,
                    SeedCount = configResults.Count,
                    WinRate = configResults.Count > 0 ? (double)successfulResults.Count / configResults.Count : 0.0
                };

                // 如果有成功结果，计算均值；否则设置为0
                if (successfulResults.Count > 0)
                {
                    aggregated.AvgTotalMoves = successfulResults.Average(r => r.TotalMoves);
                    aggregated.AvgSuccessfulGroups = successfulResults.Average(r => r.SuccessfulGroups);
                    aggregated.AvgPeakDockCount = successfulResults.Average(r => r.PeakDockCount);
                    aggregated.AvgInitialMinCost = successfulResults.Average(r => r.InitialMinCost);
                    aggregated.AvgPressureValueMean = successfulResults.Average(r => r.PressureValueMean);
                    aggregated.AvgPressureValueMin = successfulResults.Average(r => r.PressureValueMin);
                    aggregated.AvgPressureValueMax = successfulResults.Average(r => r.PressureValueMax);
                    aggregated.AvgPressureValueStdDev = successfulResults.Average(r => r.PressureValueStdDev);
                    aggregated.AvgDifficultyScore = successfulResults.Average(r => r.DifficultyScore);
                    aggregated.AvgFinalDifficulty = successfulResults.Average(r => r.FinalDifficulty);
                    aggregated.AvgEarlyPressureIndicator = successfulResults.Average(r => r.EarlyPressureIndicator);
                    aggregated.AvgTotalEarlyZeroCount = successfulResults.Average(r => r.TotalEarlyZeroCount);
                    aggregated.AvgMaxConsecutiveZeroCount = successfulResults.Average(r => r.MaxConsecutiveZeroCount);
                    aggregated.AvgConsecutiveLowPressureCount = successfulResults.Average(r => r.ConsecutiveLowPressureCount);
                    aggregated.AvgTotalEarlyLowPressureCount = successfulResults.Average(r => r.TotalEarlyLowPressureCount);
                    aggregated.AvgDifficultyPosition = successfulResults.Average(r => r.DifficultyPosition);
                }
                else
                {
                    // 所有种子都失败，设置为0
                    aggregated.AvgTotalMoves = 0;
                    aggregated.AvgSuccessfulGroups = 0;
                    aggregated.AvgPeakDockCount = 0;
                    aggregated.AvgInitialMinCost = 0;
                    aggregated.AvgPressureValueMean = 0;
                    aggregated.AvgPressureValueMin = 0;
                    aggregated.AvgPressureValueMax = 0;
                    aggregated.AvgPressureValueStdDev = 0;
                    aggregated.AvgDifficultyScore = 0;
                    aggregated.AvgFinalDifficulty = 0;
                    aggregated.AvgEarlyPressureIndicator = 0;
                    aggregated.AvgTotalEarlyZeroCount = 0;
                    aggregated.AvgMaxConsecutiveZeroCount = 0;
                    aggregated.AvgConsecutiveLowPressureCount = 0;
                    aggregated.AvgTotalEarlyLowPressureCount = 0;
                    aggregated.AvgDifficultyPosition = 0;
                }

                aggregatedResults.Add(aggregated);
            }

            Debug.Log($"聚合统计: {results.Count}条原始结果 → {aggregatedResults.Count}个配置的平均值");
            return aggregatedResults;
        }

        /// <summary>
        /// 导出结果为CSV（流式写入优化版）
        /// </summary>
        public static void ExportToCsv(List<AnalysisResult> results, string outputPath)
        {
            try
            {
                // 确保目录存在
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // 使用StreamWriter流式写入，避免StringBuilder累积大量内存
                using (var writer = new StreamWriter(outputPath, false, Encoding.UTF8, FILE_BUFFER_SIZE))
                {
                    // 写入CSV表头 - 添加难度分数字段
                    writer.WriteLine("UniqueId,TerrainId,LevelName,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,RandomSeed," +
                                    "GameCompleted,TotalMoves,GameDurationMs,CompletionStatus," +
                                    "TotalAnalysisTimeMs,SuccessfulGroups,InitialMinCost," +
                                    "DifficultyPosition,TileIdSequence,DockCountPerMove,PeakDockCount,DockAfterTrioMatch,SafeOptionCounts," +
                                    "MinCostAfterTrioMatch,MinCostOptionsAfterTrioMatch,PressureValues," +
                                    "PressureValueMean,PressureValueMin,PressureValueMax,PressureValueStdDev,DifficultyScore,FinalDifficulty," +
                                    "EarlyPressureIndicator,TotalEarlyZeroCount,MaxConsecutiveZeroCount,ConsecutiveLowPressureCount,TotalEarlyLowPressureCount,ErrorMessage");

                    // 逐行写入数据
                    foreach (var result in results)
                    {
                        string expMode = $"[{string.Join(",", result.ExperienceMode)}]";
                        string tileSequence = result.TileIdSequence.Count > 0 ? string.Join(",", result.TileIdSequence) : "";
                        string dockCounts = result.DockCountPerMove.Count > 0 ? string.Join(",", result.DockCountPerMove) : "";
                        string dockAfterTrio = result.DockAfterTrioMatch.Count > 0 ? string.Join(",", result.DockAfterTrioMatch) : "";
                        string safeOptions = result.SafeOptionCounts.Count > 0 ? string.Join(",", result.SafeOptionCounts) : "";
                        string minCostAfterTrio = result.MinCostAfterTrioMatch.Count > 0 ? string.Join(",", result.MinCostAfterTrioMatch) : "";
                        string minCostOptionsAfterTrio = result.MinCostOptionsAfterTrioMatch.Count > 0 ? string.Join(",", result.MinCostOptionsAfterTrioMatch) : "";
                        string pressureValues = result.PressureValues.Count > 1 ? string.Join(",", result.PressureValues.Take(result.PressureValues.Count - 1)) : "";

                        writer.WriteLine($"{result.UniqueId},{result.TerrainId},{result.LevelName},{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.RandomSeed}," +
                                       $"{result.GameCompleted},{result.TotalMoves},{result.GameDurationMs},\"{result.CompletionStatus}\"," +
                                       $"{result.TotalAnalysisTimeMs},{result.SuccessfulGroups},{result.InitialMinCost}," +
                                       $"{result.DifficultyPosition:F4},\"{tileSequence}\",\"{dockCounts}\",{result.PeakDockCount},\"{dockAfterTrio}\",\"{safeOptions}\"," +
                                       $"\"{minCostAfterTrio}\",\"{minCostOptionsAfterTrio}\",\"{pressureValues}\"," +
                                       $"{result.PressureValueMean:F4},{result.PressureValueMin},{result.PressureValueMax},{result.PressureValueStdDev:F4},{result.DifficultyScore:F2},{result.FinalDifficulty}," +
                                       $"{result.EarlyPressureIndicator},{result.TotalEarlyZeroCount},{result.MaxConsecutiveZeroCount},{result.ConsecutiveLowPressureCount},{result.TotalEarlyLowPressureCount},\"{result.ErrorMessage ?? ""}\"");
                    }
                } // using自动Flush和Dispose

                Debug.Log($"结果已导出到: {outputPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出CSV失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 导出聚合结果为CSV（每配置平均值）（流式写入优化版）
        /// </summary>
        public static void ExportAggregatedToCsv(List<AggregatedResult> aggregatedResults, string outputPath)
        {
            try
            {
                // 确保目录存在
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // 使用StreamWriter流式写入，避免StringBuilder累积大量内存
                using (var writer = new StreamWriter(outputPath, false, Encoding.UTF8, FILE_BUFFER_SIZE))
                {
                    // 写入CSV表头 - 23个字段
                    writer.WriteLine("TerrainId,LevelName,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,SeedCount,WinRate," +
                                    "AvgTotalMoves,AvgSuccessfulGroups,AvgPeakDockCount,AvgInitialMinCost," +
                                    "AvgPressureValueMean,AvgPressureValueMin,AvgPressureValueMax,AvgPressureValueStdDev," +
                                    "AvgDifficultyScore,AvgFinalDifficulty," +
                                    "AvgEarlyPressureIndicator,AvgTotalEarlyZeroCount,AvgMaxConsecutiveZeroCount,AvgConsecutiveLowPressureCount,AvgTotalEarlyLowPressureCount," +
                                    "AvgDifficultyPosition");

                    // 逐行写入数据
                    foreach (var result in aggregatedResults)
                    {
                        string expMode = $"[{string.Join(",", result.ExperienceMode)}]";

                        writer.WriteLine($"{result.TerrainId},{result.LevelName},{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.SeedCount},{result.WinRate:F4}," +
                                       $"{result.AvgTotalMoves:F2},{result.AvgSuccessfulGroups:F2},{result.AvgPeakDockCount:F2},{result.AvgInitialMinCost:F2}," +
                                       $"{result.AvgPressureValueMean:F4},{result.AvgPressureValueMin:F2},{result.AvgPressureValueMax:F2},{result.AvgPressureValueStdDev:F4}," +
                                       $"{result.AvgDifficultyScore:F2},{result.AvgFinalDifficulty:F2}," +
                                       $"{result.AvgEarlyPressureIndicator:F2},{result.AvgTotalEarlyZeroCount:F2},{result.AvgMaxConsecutiveZeroCount:F2},{result.AvgConsecutiveLowPressureCount:F2},{result.AvgTotalEarlyLowPressureCount:F2}," +
                                       $"{result.AvgDifficultyPosition:F4}");
                    }
                } // using自动Flush和Dispose

                Debug.Log($"聚合结果已导出到: {outputPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出聚合CSV失败: {ex.Message}");
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Unity Editor菜单：运行BattleAnalyzer批量分析（弹出配置窗口）
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/运行批量分析")]
        public static void RunBatchAnalysisFromMenu()
        {
            // 打开配置窗口
            BattleAnalyzerConfigWindow.ShowWindow();
        }

        /// <summary>
        /// Unity Editor菜单：快速运行批量分析（使用默认配置）
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/快速运行批量分析(默认配置)")]
        public static void RunBatchAnalysisFromMenuQuick()
        {
            var config = new RunConfig(); // 使用完全默认的配置
            config.OutputDirectory = Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults");

            Debug.Log($"=== 开始批量分析（默认配置） ===");
            Debug.Log($"配置详情: {config.GetConfigDescription()}");
            Debug.Log($"地形数量: {config.TestLevelCount}");

            var results = RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? $"_FixedSeeds" : "_Random";
            string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

            ExportToCsv(results, csvPath);

            // 如果启用了"仅输出每配置平均值"，生成聚合CSV
            if (config.OutputPerConfigAverage)
            {
                var aggregatedResults = AggregateResultsByConfig(results);
                var aggregatedCsvPath = csvPath.Replace(".csv", "_Aggregated.csv");
                ExportAggregatedToCsv(aggregatedResults, aggregatedCsvPath);
            }

            if (config.IsFilteringEnabled)
            {
                Debug.Log($"筛选分析完成! 找到 {results.Count} 个符合条件的结果");
            }
            else
            {
                Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
            }
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹（跨平台兼容）
            if (Directory.Exists(config.OutputDirectory))
            {
                #if UNITY_EDITOR_WIN
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
                #elif UNITY_EDITOR_OSX
                System.Diagnostics.Process.Start("open", config.OutputDirectory);
                #elif UNITY_EDITOR_LINUX
                System.Diagnostics.Process.Start("xdg-open", config.OutputDirectory);
                #else
                EditorUtility.RevealInFinder(config.OutputDirectory);
                #endif
            }
        }

    /// <summary>
    /// BattleAnalyzer配置窗口 - 用于设置RunConfig参数
    /// </summary>
    public class BattleAnalyzerConfigWindow : EditorWindow
    {
        private RunConfig config = new RunConfig();
        private Vector2 scrollPosition;
        private string seedValuesString = "";

        public static void ShowWindow()
        {
            var window = GetWindow<BattleAnalyzerConfigWindow>("BattleAnalyzer配置");
            window.minSize = new Vector2(500, 700);
            window.Show();
        }

        private void OnEnable()
        {
            // 初始化配置
            config = new RunConfig();
            config.OutputDirectory = Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults");

            // 初始化种子值字符串
            if (config.FixedSeedValues != null && config.FixedSeedValues.Length > 0)
            {
                seedValuesString = string.Join(",", config.FixedSeedValues);
            }
        }

        private void OnGUI()
        {
            scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

            // ========================================
            // 标题区域
            // ========================================
            GUILayout.Space(5);
            GUILayout.Label("BattleAnalyzer 批量分析配置", EditorStyles.boldLabel);
            GUILayout.Space(15);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ 📋 基础配置
            // ╚════════════════════════════════════════════════════════════════╝
            DrawGroupHeader("📋 基础配置");
            EditorGUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Space(8);

            // --- CSV配置模式 ---
            DrawSubHeader("CSV配置模式");
            config.ExperienceConfigEnum = EditorGUILayout.IntField(
                new GUIContent("体验模式枚举", "1-6=exp-fix-1到exp-fix-6, -1=exp-range-1所有配置, -2=数组排列组合"),
                config.ExperienceConfigEnum);
            GUILayout.Space(3);

            config.ColorCountConfigEnum = EditorGUILayout.IntField(
                new GUIContent("花色数量枚举", "1-6=type-count-1到type-count-6, -1=type-range-1所有配置, -2=动态范围"),
                config.ColorCountConfigEnum);

            // --- 排列组合配置（条件显示） ---
            if (config.ExperienceConfigEnum == -2)
            {
                GUILayout.Space(10);
                DrawSubHeader("排列组合配置");
                EditorGUI.indentLevel++;

                config.ArrayLength = EditorGUILayout.IntField(
                    new GUIContent("数组长度", "体验数组长度，如[a,b,c]为3"),
                    config.ArrayLength);
                GUILayout.Space(3);

                config.MinValue = EditorGUILayout.IntField(
                    new GUIContent("最小值", "排列组合的最小值"),
                    config.MinValue);
                GUILayout.Space(3);

                config.MaxValue = EditorGUILayout.IntField(
                    new GUIContent("最大值", "排列组合的最大值"),
                    config.MaxValue);

                EditorGUI.indentLevel--;
            }

            // --- 测试地形数量 ---
            GUILayout.Space(10);
            DrawSubHeader("测试范围");
            config.TestLevelCount = EditorGUILayout.IntField(
                new GUIContent("测试地形数量", "要测试的地形数量"),
                config.TestLevelCount);

            GUILayout.Space(8);
            EditorGUILayout.EndVertical();
            GUILayout.Space(15);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ 🎲 随机控制
            // ╚════════════════════════════════════════════════════════════════╝
            DrawGroupHeader("🎲 随机控制");
            EditorGUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Space(8);

            // --- 配置选择策略 ---
            DrawSubHeader("配置选择策略");
            config.UseRandomConfigSelection = EditorGUILayout.Toggle(
                new GUIContent("随机选择配置", "true=在范围内随机选择配置（不重复），false=按顺序遍历配置"),
                config.UseRandomConfigSelection);

            // --- 种子配置 ---
            GUILayout.Space(10);
            DrawSubHeader("种子配置");
            config.UseFixedSeed = EditorGUILayout.Toggle(
                new GUIContent("使用固定种子", "true=结果可重现，false=完全随机"),
                config.UseFixedSeed);
            GUILayout.Space(5);

            EditorGUILayout.LabelField("固定种子值列表（逗号分隔）:");
            seedValuesString = EditorGUILayout.TextArea(seedValuesString, GUILayout.Height(40));
            GUILayout.Space(5);

            config.MaxSeedAttemptsPerConfig = EditorGUILayout.IntField(
                new GUIContent("每配置最大种子尝试次数", "筛选模式下，每个配置最多尝试多少个种子"),
                config.MaxSeedAttemptsPerConfig);
            GUILayout.Space(3);

            config.MaxEmptySeedAttemptsPerConfig = EditorGUILayout.IntField(
                new GUIContent("每配置最大空运行次数", "当配置连续尝试x次都找不到符合条件的种子时提前退出"),
                config.MaxEmptySeedAttemptsPerConfig);

            GUILayout.Space(8);
            EditorGUILayout.EndVertical();
            GUILayout.Space(15);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ 🔍 筛选条件
            // ╚════════════════════════════════════════════════════════════════╝
            DrawGroupHeader("🔍 筛选条件");
            EditorGUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Space(8);

            // --- 筛选模式开关 ---
            DrawSubHeader("筛选模式开关");
            config.UseTerrainSpecificFiltering = EditorGUILayout.Toggle(
                new GUIContent("使用地形特定筛选", "从CSV读取position和score字段"),
                config.UseTerrainSpecificFiltering);
            GUILayout.Space(3);

            config.EnableGlobalFiltering = EditorGUILayout.Toggle(
                new GUIContent("启用全局筛选", "作为fallback使用"),
                config.EnableGlobalFiltering);
            GUILayout.Space(3);

            config.UseAverageFiltering = EditorGUILayout.Toggle(
                new GUIContent("使用平均值筛选", "true=跑满种子后对平均值筛选，false=每个种子立即筛选"),
                config.UseAverageFiltering);

            // --- 全局筛选范围（条件显示） ---
            if (config.EnableGlobalFiltering)
            {
                GUILayout.Space(10);
                DrawSubHeader("全局筛选范围");
                EditorGUI.indentLevel++;

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("难点位置 (Pos)", GUILayout.Width(140));
                config.GlobalDifficultyPositionRangeMin = EditorGUILayout.FloatField(config.GlobalDifficultyPositionRangeMin, GUILayout.Width(60));
                EditorGUILayout.LabelField("~", GUILayout.Width(15));
                config.GlobalDifficultyPositionRangeMax = EditorGUILayout.FloatField(config.GlobalDifficultyPositionRangeMax, GUILayout.Width(60));
                EditorGUILayout.EndHorizontal();
                GUILayout.Space(3);

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("难度分数 (Score)", GUILayout.Width(140));
                config.GlobalDifficultyScoreRangeMin = EditorGUILayout.FloatField(config.GlobalDifficultyScoreRangeMin, GUILayout.Width(60));
                EditorGUILayout.LabelField("~", GUILayout.Width(15));
                config.GlobalDifficultyScoreRangeMax = EditorGUILayout.FloatField(config.GlobalDifficultyScoreRangeMax, GUILayout.Width(60));
                EditorGUILayout.EndHorizontal();
                GUILayout.Space(3);

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("连续低压力 (ConLP)", GUILayout.Width(140));
                config.GlobalConsecutiveLowPressureRangeMin = EditorGUILayout.IntField(config.GlobalConsecutiveLowPressureRangeMin, GUILayout.Width(60));
                EditorGUILayout.LabelField("~", GUILayout.Width(15));
                config.GlobalConsecutiveLowPressureRangeMax = EditorGUILayout.IntField(config.GlobalConsecutiveLowPressureRangeMax, GUILayout.Width(60));
                EditorGUILayout.EndHorizontal();
                GUILayout.Space(3);

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField("前期低压力总数 (TotLP)", GUILayout.Width(140));
                config.GlobalTotalEarlyLowPressureRangeMin = EditorGUILayout.IntField(config.GlobalTotalEarlyLowPressureRangeMin, GUILayout.Width(60));
                EditorGUILayout.LabelField("~", GUILayout.Width(15));
                config.GlobalTotalEarlyLowPressureRangeMax = EditorGUILayout.IntField(config.GlobalTotalEarlyLowPressureRangeMax, GUILayout.Width(60));
                EditorGUILayout.EndHorizontal();

                EditorGUI.indentLevel--;
            }

            GUILayout.Space(8);
            EditorGUILayout.EndVertical();
            GUILayout.Space(15);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ ⚙️ 执行控制 & 输出
            // ╚════════════════════════════════════════════════════════════════╝
            DrawGroupHeader("⚙️ 执行控制 & 输出");
            EditorGUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Space(8);

            // --- 执行控制参数 ---
            DrawSubHeader("执行控制参数");
            config.RequiredResultsPerTerrain = EditorGUILayout.IntField(
                new GUIContent("每地形需要结果数量", "每个地形需要找到的符合条件结果数量"),
                config.RequiredResultsPerTerrain);
            GUILayout.Space(3);

            config.MaxConfigAttemptsPerTerrain = EditorGUILayout.IntField(
                new GUIContent("每地形最大尝试配置数量", "每个地形最多尝试多少个配置组合"),
                config.MaxConfigAttemptsPerTerrain);

            // --- 输出配置 ---
            GUILayout.Space(10);
            DrawSubHeader("输出配置");
            EditorGUILayout.LabelField("输出目录:");
            config.OutputDirectory = EditorGUILayout.TextField(config.OutputDirectory);
            GUILayout.Space(5);

            if (GUILayout.Button("选择输出目录"))
            {
                string selectedPath = EditorUtility.OpenFolderPanel("选择输出目录", config.OutputDirectory, "");
                if (!string.IsNullOrEmpty(selectedPath))
                {
                    config.OutputDirectory = selectedPath;
                }
            }
            GUILayout.Space(5);

            config.OutputPerConfigAverage = EditorGUILayout.Toggle(
                new GUIContent("仅输出每配置平均值", "开启后会额外生成_Aggregated.csv文件，每条配置输出一行平均值（同地形、同体验模式、同花色数量的所有种子的平均值）"),
                config.OutputPerConfigAverage);

            GUILayout.Space(8);
            EditorGUILayout.EndVertical();
            GUILayout.Space(15);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ 📊 配置预览
            // ╚════════════════════════════════════════════════════════════════╝
            DrawGroupHeader("📊 配置预览");
            EditorGUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Space(8);

            EditorGUILayout.LabelField("配置描述:", EditorStyles.wordWrappedLabel);
            GUILayout.Space(3);
            EditorGUILayout.LabelField(config.GetConfigDescription(), EditorStyles.wordWrappedMiniLabel);

            GUILayout.Space(8);
            EditorGUILayout.EndVertical();
            GUILayout.Space(20);

            // ╔════════════════════════════════════════════════════════════════╗
            // ║ ▶️ 执行操作
            // ╚════════════════════════════════════════════════════════════════╝
            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("▶️ 运行批量分析", GUILayout.Height(40)))
            {
                RunAnalysis();
            }

            if (GUILayout.Button("🔄 重置为默认配置", GUILayout.Height(40)))
            {
                ResetToDefault();
            }

            EditorGUILayout.EndHorizontal();
            GUILayout.Space(10);

            EditorGUILayout.EndScrollView();
        }

        /// <summary>
        /// 绘制主分组标题（带图标）
        /// </summary>
        private void DrawGroupHeader(string title)
        {
            GUIStyle headerStyle = new GUIStyle(EditorStyles.boldLabel)
            {
                fontSize = 13,
                padding = new RectOffset(0, 0, 5, 5)
            };
            GUILayout.Label(title, headerStyle);
            GUILayout.Space(5);
        }

        /// <summary>
        /// 绘制子分组标题（灰色小标题）
        /// </summary>
        private void DrawSubHeader(string title)
        {
            GUIStyle subHeaderStyle = new GUIStyle(EditorStyles.miniLabel)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = new Color(0.6f, 0.6f, 0.6f) }
            };
            GUILayout.Label(title, subHeaderStyle);
            GUILayout.Space(3);
        }

        private void RunAnalysis()
        {
            // 解析种子值字符串
            ParseSeedValues();

            // 确保输出目录存在
            if (!Directory.Exists(config.OutputDirectory))
            {
                Directory.CreateDirectory(config.OutputDirectory);
            }

            Debug.Log($"=== 开始批量分析（自定义配置） ===");
            Debug.Log($"配置详情: {config.GetConfigDescription()}");
            Debug.Log($"地形数量: {config.TestLevelCount}");

            var results = BattleAnalyzerRunner.RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? $"_FixedSeeds" : "_Random";
            string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

            BattleAnalyzerRunner.ExportToCsv(results, csvPath);

            // 如果启用了"仅输出每配置平均值"，生成聚合CSV
            if (config.OutputPerConfigAverage)
            {
                var aggregatedResults = BattleAnalyzerRunner.AggregateResultsByConfig(results);
                var aggregatedCsvPath = csvPath.Replace(".csv", "_Aggregated.csv");
                BattleAnalyzerRunner.ExportAggregatedToCsv(aggregatedResults, aggregatedCsvPath);
            }

            if (config.IsFilteringEnabled)
            {
                Debug.Log($"筛选分析完成! 找到 {results.Count} 个符合条件的结果");
            }
            else
            {
                Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
            }
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹（跨平台兼容）
            if (Directory.Exists(config.OutputDirectory))
            {
                #if UNITY_EDITOR_WIN
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
                #elif UNITY_EDITOR_OSX
                System.Diagnostics.Process.Start("open", config.OutputDirectory);
                #elif UNITY_EDITOR_LINUX
                System.Diagnostics.Process.Start("xdg-open", config.OutputDirectory);
                #else
                EditorUtility.RevealInFinder(config.OutputDirectory);
                #endif
            }

            // 关闭窗口
            Close();
        }

        private void ParseSeedValues()
        {
            if (string.IsNullOrWhiteSpace(seedValuesString))
            {
                config.FixedSeedValues = new int[] { 12345678, 11111111, 22222222, 33333333, 44444444, 55555555, 66666666, 77777777, 88888888, 99999999 };
                return;
            }

            try
            {
                var parts = seedValuesString.Split(',');
                var seedList = new List<int>();

                foreach (var part in parts)
                {
                    if (int.TryParse(part.Trim(), out int seed))
                    {
                        seedList.Add(seed);
                    }
                }

                config.FixedSeedValues = seedList.Count > 0 ? seedList.ToArray() : new int[] { 12345678 };
            }
            catch
            {
                config.FixedSeedValues = new int[] { 12345678 };
            }
        }

        private void ResetToDefault()
        {
            config = new RunConfig();
            config.OutputDirectory = Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults");

            if (config.FixedSeedValues != null && config.FixedSeedValues.Length > 0)
            {
                seedValuesString = string.Join(",", config.FixedSeedValues);
            }
        }
    }

#endif
    }
}