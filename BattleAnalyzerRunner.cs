// ========== 日志级别控制 ==========
// 通过定义符号控制详细日志输出程度
// Unity菜单: Edit → Project Settings → Player → Scripting Define Symbols
// 添加 VERBOSE_ANALYZER_LOGGING 启用详细日志(每个种子尝试都输出)
// 默认:NORMAL_ANALYZER_LOGGING(每100个任务输出进度)
#if !VERBOSE_ANALYZER_LOGGING
#define NORMAL_ANALYZER_LOGGING
#endif

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using UnityEngine;
using DGuo.Client.TileMatch;
using DGuo.Client;
using DGuo.Client.TileMatch.DesignerAlgo.Core;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DGuo.Client.TileMatch.Analysis
{
    /// <summary>
    /// BattleAnalyzer自动运行器 - 读取all_level.csv配置,批量生成关卡并分析
    /// </summary>
    public class BattleAnalyzerRunner
    {
        // ========== 常量定义 ==========
        private const int FILE_BUFFER_SIZE = 65536; // 64KB文件IO缓冲区
        private const int CSV_PARSER_BUFFER_SIZE = 256; // CSV解析器StringBuilder初始容量

        // ========== 数据结构 ==========

        /// <summary>
        /// CSV配置行数据
        /// </summary>
        public class CsvLevelConfig
        {
            public int RowIndex { get; set; } // CSV行索引(从0开始,0=表头后第一行)
            public int TerrainId { get; set; }
            public int[] ExpFix1 { get; set; }
            public int[] ExpRange1 { get; set; }
            public int TypeCount1 { get; set; }
            public int TypeRange1 { get; set; }
        }

        /// <summary>
        /// 自动游戏分析结果
        /// </summary>
        public class AnalysisResult
        {
            public string UniqueId { get; set; } // 唯一标识符
            public string BatchName { get; set; } // 批次名称(多批筛选时用于标识)
            public int RowIndex { get; set; } // CSV行索引
            public int TerrainId { get; set; }
            public string LevelName { get; set; }
            public string AlgorithmName { get; set; } // 生成算法版本名
            public int[] ExperienceMode { get; set; }
            public int ColorCount { get; set; }
            public int TotalTiles { get; set; }
            public int RandomSeed { get; set; } // 随机种子

            // 游戏执行结果
            public bool GameCompleted { get; set; }
            public int GameDurationMs { get; set; }
            public string CompletionStatus { get; set; }

            // 关键快照数据
            public int PeakDockCount { get; set; }
            public int MinMovesToComplete { get; set; }
            public int InitialMinCost { get; set; } // 游戏开局时的最小cost值
            public double DifficultyPosition { get; set; } // 难点位置:0~1,表示peakdock在关卡进度中的位置
            public List<int> PressureValues { get; set; } = new List<int>(); // 压力值列表:开局+每次三消后

            // 压力值统计字段
            public double PressureValueMean { get; set; } // 压力值均值
            public int PressureValueMin { get; set; } // 压力值最小值
            public int PressureValueMax { get; set; } // 压力值最大值
            public double PressureValueStdDev { get; set; } // 压力值标准差
            public double DifficultyScore { get; set; } // 难度分数:(0.5*均值/5+0.3*标准差/2+0.2*最大值/5)*500
            public int FinalDifficulty { get; set; } // 最终难度:1-5
            public int ConsecutiveLowPressureCount { get; set; } // PressureValues从第一个开始连续1的数量
            public int TotalEarlyLowPressureCount { get; set; } // PressureValues前7个中1的总数

            public string ErrorMessage { get; set; }
        }

        /// <summary>
        /// 配置聚合键 - 用于分组同一配置的多个种子结果
        /// </summary>
        public class ConfigKey : IEquatable<ConfigKey>
        {
            public int TerrainId { get; set; }
            public string ExperienceModeStr { get; set; } // "[1,2,3]"格式
            public int ColorCount { get; set; }

            public override bool Equals(object obj) => Equals(obj as ConfigKey);

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
            public string BatchName { get; set; } // 批次名称(聚合也按批次分组)
            public int[] ExperienceMode { get; set; }
            public int ColorCount { get; set; }
            public int TotalTiles { get; set; }
            public string AlgorithmName { get; set; }

            // 种子统计
            public int SeedCount { get; set; } // 运行的种子数量(包含成功和失败)
            public List<int> SeedList { get; set; } = new List<int>(); // 使用的种子列表
            public double WinRate { get; set; } // 胜率:成功通关的种子占总种子的百分比(0.0-1.0)

            // 压力分析均值(基于成功通关的种子计算)
            public double AvgPeakDockCount { get; set; }
            public double AvgInitialMinCost { get; set; }
            public double AvgPressureValueMean { get; set; }
            public double AvgPressureValueMin { get; set; }
            public double AvgPressureValueMax { get; set; }
            public double AvgPressureValueStdDev { get; set; }
            public double AvgDifficultyScore { get; set; }
            public double AvgFinalDifficulty { get; set; }
            public double AvgConsecutiveLowPressureCount { get; set; }
            public double AvgTotalEarlyLowPressureCount { get; set; }
            public double AvgDifficultyPosition { get; set; }
        }

        /// <summary>
        /// 批量运行配置 - 简化版本
        /// </summary>
        [System.Serializable]
        public class RunConfig
        {
            [Header("=== CSV配置选择器 ===")]
            public int ExperienceConfigEnum = -2; // 体验模式枚举:1=exp-fix-1, -1=exp-range-1所有配置, -2=数组排列组合
            public int ColorCountConfigEnum = -2; // 花色数量枚举:1=type-count-1, -1=type-range-1所有配置, -2=动态范围

            [Header("=== 测试参数 ===")]
            public int TestLevelCount = 5; // 测试地形数量

            [Header("=== 排列组合配置 (ExperienceConfigEnum = -2时生效) ===")]
            public int ArrayLength = 3; // 数组长度
            public int MinValue = 1; // 最小值
            public int MaxValue = 9; // 最大值

            [Header("=== 配置选择策略 ===")]
            public bool UseRandomConfigSelection = true; // 是否随机选择配置:true=在范围内随机选择(不重复),false=按顺序遍历

            [Header("=== 随机种子配置 ===")]
            public bool UseFixedSeed = false; // 是否使用固定种子:true=结果可重现,false=完全随机
            public int[] FixedSeedValues = { 12345678, 11111111, 22222222, 33333333, 44444444, 55555555, 66666666, 77777777, 88888888, 99999999 }; // 固定种子值列表
            public int MaxSeedAttemptsPerConfig = 1000; // 每个配置最大种子尝试次数:在筛选模式下用于搜索符合条件的种子
            public int MaxEmptySeedAttemptsPerConfig = 100; // 每配置最大空运行次数:当配置连续x次都找不到符合条件的种子时提前退出

            [Header("=== 输出配置 ===")]
            public string OutputDirectory = "BattleAnalysisResults";
            public bool OutputPerConfigAverage = true; // 是否仅输出每配置平均值(同地形、同体验模式、同花色数量的所有种子的平均值)

            [Header("=== 并行配置 ===")]
            public bool EnableParallel = false; // 是否按地形行进行并行
            public int MaxParallelRows = 2; // 最大并发的地形行数

            [Header("=== 筛选配置 ===")]
            public bool EnableGlobalFiltering = false; // 是否启用全局筛选
            public bool UseAverageFiltering = false; // 是否使用平均值筛选:true=跑满种子后对平均值筛选,false=每个种子立即筛选
            public float GlobalDifficultyPositionRangeMin = 0.5f; // 全局难点位置范围最小值
            public float GlobalDifficultyPositionRangeMax = 0.99f; // 全局难点位置范围最大值
            public float GlobalDifficultyScoreRangeMin = 150f; // 全局难度分数范围最小值
            public float GlobalDifficultyScoreRangeMax = 300f; // 全局难度分数范围最大值
            public int GlobalConsecutiveLowPressureRangeMin = 0; // 全局连续低压力范围最小值
            public int GlobalConsecutiveLowPressureRangeMax = 100; // 全局连续低压力范围最大值
            public int GlobalTotalEarlyLowPressureRangeMin = 4; // 全局前期低压力总数范围最小值
            public int GlobalTotalEarlyLowPressureRangeMax = 7; // 全局前期低压力总数范围最大值
            public int RequiredResultsPerTerrain = 1; // 每个地形需要找到的符合条件结果数量
            public int MaxConfigAttemptsPerTerrain = 1000; // 每个地形最大尝试配置数量

            [Header("=== 批次筛选配置(可选, 配置后启用多批筛选) ===")]
            public List<BatchFilter> BatchFilters = new List<BatchFilter>();

            /// <summary>
            /// 检查聚合结果是否符合筛选条件(用于平均值筛选模式)
            /// </summary>
            public bool MatchesCriteria(AggregatedResult result)
            {
                if (!EnableGlobalFiltering) return true;

                return result.AvgDifficultyPosition >= GlobalDifficultyPositionRangeMin &&
                       result.AvgDifficultyPosition <= GlobalDifficultyPositionRangeMax &&
                       result.AvgDifficultyScore >= GlobalDifficultyScoreRangeMin &&
                       result.AvgDifficultyScore <= GlobalDifficultyScoreRangeMax &&
                       result.AvgConsecutiveLowPressureCount >= GlobalConsecutiveLowPressureRangeMin &&
                       result.AvgConsecutiveLowPressureCount <= GlobalConsecutiveLowPressureRangeMax &&
                       result.AvgTotalEarlyLowPressureCount >= GlobalTotalEarlyLowPressureRangeMin &&
                       result.AvgTotalEarlyLowPressureCount <= GlobalTotalEarlyLowPressureRangeMax;
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
                    return 12345678 + terrainId * 10000 + seedAttemptIndex;
                }
                // 非固定种子: 使用线程本地的System.Random避免并发冲突
                return _threadRandom.Value.Next(1, int.MaxValue);
            }

            /// <summary>
            /// 检查分析结果是否符合筛选条件 (单次结果筛选)
            /// 注意: 当EnableGlobalFiltering为true时, 非平均值筛选模式下只有成功通关的关卡才会进入此判断
            /// </summary>
            public bool MatchesCriteria(AnalysisResult result)
            {
                if (!EnableGlobalFiltering) return true;

                return result.DifficultyPosition >= GlobalDifficultyPositionRangeMin &&
                       result.DifficultyPosition <= GlobalDifficultyPositionRangeMax &&
                       result.DifficultyScore >= GlobalDifficultyScoreRangeMin &&
                       result.DifficultyScore <= GlobalDifficultyScoreRangeMax &&
                       result.ConsecutiveLowPressureCount >= GlobalConsecutiveLowPressureRangeMin &&
                       result.ConsecutiveLowPressureCount <= GlobalConsecutiveLowPressureRangeMax &&
                       result.TotalEarlyLowPressureCount >= GlobalTotalEarlyLowPressureRangeMin &&
                       result.TotalEarlyLowPressureCount <= GlobalTotalEarlyLowPressureRangeMax;
            }

            /// <summary>
            /// 检查是否启用了任何筛选
            /// </summary>
            public bool IsFilteringEnabled => EnableGlobalFiltering;

            /// <summary>
            /// 获取筛选条件描述(简洁格式)
            /// </summary>
            public string GetFilterDescription()
            {
                if (!IsFilteringEnabled) return "筛选已禁用";

                var ranges = new List<string>
                {
                    $"Pos[{GlobalDifficultyPositionRangeMin:F2}~{GlobalDifficultyPositionRangeMax:F2}]",
                    $"Score[{GlobalDifficultyScoreRangeMin:F0}~{GlobalDifficultyScoreRangeMax:F0}]",
                    $"ConLP[{GlobalConsecutiveLowPressureRangeMin}~{GlobalConsecutiveLowPressureRangeMax}]",
                    $"TotLP[{GlobalTotalEarlyLowPressureRangeMin}~{GlobalTotalEarlyLowPressureRangeMax}]"
                };
                return $"全局筛选: {string.Join(" | ", ranges)}, 每地形需要{RequiredResultsPerTerrain}个结果";
            }

            /// <summary>
            /// 获取配置描述信息
            /// </summary>
            public string GetConfigDescription()
            {
                string expMode = ExperienceConfigEnum switch
                {
                    1 => "ExpFix1",
                    -1 => "所有ExpRange1配置",
                    -2 => $"排列组合[{MinValue}-{MaxValue}]^{ArrayLength}",
                    _ => $"配置{ExperienceConfigEnum}"
                };

                string colorMode = ColorCountConfigEnum switch
                {
                    1 => "TypeCount1",
                    -1 => "所有TypeRange1配置",
                    -2 => "动态花色范围(总tile数/3的40%-80%,上限25)",
                    _ => $"配置{ColorCountConfigEnum}"
                };

                string seedMode = UseFixedSeed ? $"固定种子列表({FixedSeedValues?.Length ?? 0}个)" : "随机种子";
                string multiBatch = (BatchFilters != null && BatchFilters.Count > 0) ? $", 多批筛选({BatchFilters.Count}批)" : "";
                string filterMode = IsFilteringEnabled ? $", 筛选[{GetFilterDescription()}]" : "";
                string attempts = $", 最多尝试{MaxConfigAttemptsPerTerrain}个配置, 每配置最多{MaxSeedAttemptsPerConfig}个种子";

                return $"体验模式[{expMode}], 花色数量[{colorMode}], {seedMode}{multiBatch}{filterMode}{attempts}";
            }

            /// <summary>
            /// 批次筛选配置
            /// </summary>
            [System.Serializable]
            public class BatchFilter
            {
                public string BatchName = "Batch-1";
                public float DifficultyPositionRangeMin = 0.5f;
                public float DifficultyPositionRangeMax = 0.99f;
                public float DifficultyScoreRangeMin = 150f;
                public float DifficultyScoreRangeMax = 300f;
                public int ConsecutiveLowPressureRangeMin = 0;
                public int ConsecutiveLowPressureRangeMax = 100;
                public int TotalEarlyLowPressureRangeMin = 4;
                public int TotalEarlyLowPressureRangeMax = 7;

                /// <summary>
                /// 检查单次分析结果是否匹配此批次的筛选条件
                /// 注意: 调用此方法前应先检查result.GameCompleted (非平均值筛选模式)
                /// </summary>
                public bool Matches(AnalysisResult result)
                {
                    return result.DifficultyPosition >= DifficultyPositionRangeMin &&
                           result.DifficultyPosition <= DifficultyPositionRangeMax &&
                           result.DifficultyScore >= DifficultyScoreRangeMin &&
                           result.DifficultyScore <= DifficultyScoreRangeMax &&
                           result.ConsecutiveLowPressureCount >= ConsecutiveLowPressureRangeMin &&
                           result.ConsecutiveLowPressureCount <= ConsecutiveLowPressureRangeMax &&
                           result.TotalEarlyLowPressureCount >= TotalEarlyLowPressureRangeMin &&
                           result.TotalEarlyLowPressureCount <= TotalEarlyLowPressureRangeMax;
                }

                /// <summary>
                /// 检查聚合结果是否匹配此批次的筛选条件 (平均值筛选模式)
                /// </summary>
                public bool Matches(AggregatedResult result)
                {
                    return result.AvgDifficultyPosition >= DifficultyPositionRangeMin &&
                           result.AvgDifficultyPosition <= DifficultyPositionRangeMax &&
                           result.AvgDifficultyScore >= DifficultyScoreRangeMin &&
                           result.AvgDifficultyScore <= DifficultyScoreRangeMax &&
                           result.AvgConsecutiveLowPressureCount >= ConsecutiveLowPressureRangeMin &&
                           result.AvgConsecutiveLowPressureCount <= ConsecutiveLowPressureRangeMax &&
                           result.AvgTotalEarlyLowPressureCount >= TotalEarlyLowPressureRangeMin &&
                           result.AvgTotalEarlyLowPressureCount <= TotalEarlyLowPressureRangeMax;
                }
            }
        }

        // ========== 静态缓存变量 ==========
        private static List<CsvLevelConfig> _csvConfigs = null;
        private static readonly object _csvLock = new object();
        private static Dictionary<int, LevelData> _levelDataCache = new Dictionary<int, LevelData>();
        private static readonly object _levelDataCacheLock = new object();
        private static readonly Dictionary<int, List<int>> _standardColorsCache = new Dictionary<int, List<int>>();
        private static List<int> _colorPoolCache = null; // 花色池缓存
        private static readonly System.Threading.ThreadLocal<System.Random> _threadRandom =
            new System.Threading.ThreadLocal<System.Random>(() =>
                new System.Random(unchecked(Environment.TickCount * 31 + System.Threading.Thread.CurrentThread.ManagedThreadId)));
        
        private static int _globalUniqueIdCounter = 0; // 并发下生成唯一id计数

        // ========== CSV配置管理器 ==========

        /// <summary>
        /// CSV配置管理器 - 统一的配置加载和解析服务
        /// </summary>
        public static class CsvConfigManager
        {
            /// <summary>
            /// 加载CSV配置数据 - 线程安全优化版本,按行存储
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
                        string csvPath = Path.Combine(Application.dataPath, "验证器", "Editor", "all_level.csv");

                        if (!File.Exists(csvPath))
                        {
                            Debug.LogError($"[BattleAnalyzer] CSV配置文件不存在: {csvPath}");
                            return;
                        }

                        Debug.Log($"[BattleAnalyzer] 开始加载CSV配置: {csvPath}");

                        int successCount = 0;
                        int failedCount = 0;

                        using (var fileStream = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.Read, FILE_BUFFER_SIZE))
                        using (var reader = new StreamReader(fileStream, Encoding.UTF8, true, FILE_BUFFER_SIZE))
                        {
                            string headerLine = reader.ReadLine(); // 跳过表头

                            string line;
                            int rowIndex = 0; // 行索引,从0开始

                            while ((line = reader.ReadLine()) != null)
                            {
                                var parts = CsvParser.ParseCsvLine(line);
                                if (parts.Length >= 10 && int.TryParse(parts[0], out int terrainId))
                                {
                                    var config = new CsvLevelConfig
                                    {
                                        RowIndex = rowIndex,
                                        TerrainId = terrainId,
                                        ExpFix1 = CsvParser.ParseIntArray(parts[1]),
                                        ExpRange1 = CsvParser.ParseIntArray(parts[7]),
                                        TypeCount1 = CsvParser.ParseIntOrDefault(parts[8], 1),
                                        TypeRange1 = CsvParser.ParseIntOrDefault(parts[14], 1)
                                    };
                                    _csvConfigs.Add(config);
                                    successCount++;
                                    rowIndex++;
                                }
                                else
                                {
                                    failedCount++;
                                    Debug.LogWarning($"[BattleAnalyzer] CSV行{rowIndex + 2}解析失败");
                                }
                            }
                        }

                        Debug.Log($"[BattleAnalyzer] CSV加载完成: 成功{successCount}行, 失败{failedCount}行, 总计{_csvConfigs.Count}行配置");
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"[BattleAnalyzer] 加载CSV配置失败: {ex.Message}");
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

                if (rowIndex < 0 || rowIndex >= _csvConfigs.Count)
                {
                    Debug.LogWarning($"行索引 {rowIndex} 超出范围,使用默认值");
                    return new int[][] { new int[] { 1, 2, 3 } };
                }

                var config = _csvConfigs[rowIndex];

                return experienceConfigEnum switch
                {
                    1 => new int[][] { config.ExpFix1 },
                    -1 => GetAllExpRange1Configurations(),
                    -2 when runConfig != null => GeneratePermutations(runConfig.ArrayLength, runConfig.MinValue, runConfig.MaxValue),
                    _ => new int[][] { new int[] { 1, 2, 3 } }
                };
            }

            /// <summary>
            /// 生成排列组合:从minValue到maxValue的所有长度为arrayLength的组合
            /// </summary>
            private static int[][] GeneratePermutations(int arrayLength, int minValue, int maxValue)
            {
                int valueRange = maxValue - minValue + 1;
                int totalPermutations = (int)Math.Pow(valueRange, arrayLength);

                var permutations = new List<int[]>();

                for (int i = 0; i < totalPermutations; i++)
                {
                    var permutation = new int[arrayLength];
                    int temp = i;

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

                if (rowIndex < 0 || rowIndex >= _csvConfigs.Count)
                {
                    Debug.LogWarning($"行索引 {rowIndex} 超出范围,使用默认值");
                    return new int[] { 7 };
                }

                var config = _csvConfigs[rowIndex];

                return colorCountConfigEnum switch
                {
                    1 => new int[] { config.TypeCount1 },
                    -1 => GetAllTypeRange1Configurations(),
                    -2 => GenerateDynamicColorRange(config.TerrainId),
                    _ => new int[] { 7 }
                };
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
            /// 生成基于总tile数的动态花色范围:总tile数/3的40%-80%,向下取整,上限25
            /// </summary>
            private static int[] GenerateDynamicColorRange(int terrainId)
            {
                try
                {
                    var levelData = LoadLevelData(terrainId.ToString());
                    if (levelData == null)
                    {
                        Debug.LogWarning($"无法加载关卡 {terrainId} 数据,使用默认范围");
                        return new int[] { 5, 6, 7, 8, 9, 10, 11, 12 };
                    }

                    int totalTiles = CalculateTotalTileCount(levelData);
                    int totalGroups = totalTiles / 3;

                    int minColorCount = Mathf.FloorToInt(totalGroups * 0.4f);
                    int maxColorCount = Mathf.FloorToInt(totalGroups * 0.8f);

                    minColorCount = Math.Max(1, minColorCount);
                    maxColorCount = Math.Min(25, Math.Max(minColorCount, maxColorCount));

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
                    return new int[] { 7 };
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
        }

        // ========== CSV解析工具类 ==========

        /// <summary>
        /// CSV解析工具类 - 提取通用解析逻辑,优化性能
        /// </summary>
        public static class CsvParser
        {
            private static readonly StringBuilder _reusableStringBuilder = new StringBuilder(CSV_PARSER_BUFFER_SIZE);

            /// <summary>
            /// 解析CSV行,处理引号包围的字段
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
            /// 解析整数数组字符串
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
                        return result.ToArray();
                }
                catch
                {
                    return new int[] { 1, 2, 3 };
                }
            }

            /// <summary>
            /// 解析整数或返回默认值
            /// </summary>
            public static int ParseIntOrDefault(string str, int defaultValue)
            {
                return string.IsNullOrEmpty(str) || !int.TryParse(str.Trim(), out int result) ? defaultValue : result;
            }
        }

        // ========== 核心分析方法 ==========

        /// <summary>
        /// 运行单个关卡分析
        /// </summary>
        public static AnalysisResult RunSingleLevelAnalysis(string levelName, int[] experienceMode, int colorCount, int randomSeed = -1)
        {
            // 如果没有指定种子,生成随机种子
            if (randomSeed == -1)
            {
                randomSeed = _threadRandom.Value.Next(1, int.MaxValue);
            }

            // 注意: 不再使用UnityEngine.Random全局状态

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
                // 基于(seed, terrainId, colorCount)的确定性洗牌,保证并发与串行一致
                int levelIdForColorSeed = 0; int.TryParse(levelName, out levelIdForColorSeed);
                var availableColors = CreateAvailableColorsDeterministic(colorCount, randomSeed, levelIdForColorSeed);

                // 使用RuleBasedAlgorithm进行花色分配(内部已自动执行虚拟游戏模拟)
                var algorithm = new DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo.RuleBasedAlgorithm();
                algorithm.InitializeRandomSeed(randomSeed);
                algorithm.AssignTileTypes(tiles, experienceMode, availableColors);

                // 获取真实使用的算法名称
                result.AlgorithmName = algorithm.AlgorithmName;

                // 3. 直接从算法获取动态复杂度评估结果(无需再次运行虚拟游戏)
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
                result.CompletionStatus = dynamicResult.CompletionStatus;
                result.PeakDockCount = dynamicResult.PeakDockCount;
                result.MinMovesToComplete = dynamicResult.MinMovesToComplete;
                result.InitialMinCost = dynamicResult.InitialMinCost;
                result.DifficultyPosition = dynamicResult.DifficultyPosition;
                result.PressureValues = dynamicResult.PressureValues;
                result.PressureValueMean = dynamicResult.PressureValueMean;
                result.PressureValueMin = dynamicResult.PressureValueMin;
                result.PressureValueMax = dynamicResult.PressureValueMax;
                result.PressureValueStdDev = dynamicResult.PressureValueStdDev;
                result.DifficultyScore = dynamicResult.DifficultyScore;
                result.FinalDifficulty = dynamicResult.FinalDifficulty;
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

        // ========== 辅助方法 ==========

        /// <summary>
        /// 基于原始结果创建带批次标识的拷贝(避免重复生成关卡配置)
        /// </summary>
        private static AnalysisResult CloneForBatch(AnalysisResult src, string batchName, string uniqueId)
        {
            return new AnalysisResult
            {
                UniqueId = uniqueId,
                BatchName = batchName,
                RowIndex = src.RowIndex,
                TerrainId = src.TerrainId,
                LevelName = src.LevelName,
                AlgorithmName = src.AlgorithmName,
                ExperienceMode = src.ExperienceMode != null ? (int[])src.ExperienceMode.Clone() : null,
                ColorCount = src.ColorCount,
                TotalTiles = src.TotalTiles,
                RandomSeed = src.RandomSeed,
                GameCompleted = src.GameCompleted,
                GameDurationMs = src.GameDurationMs,
                CompletionStatus = src.CompletionStatus,
                PeakDockCount = src.PeakDockCount,
                MinMovesToComplete = src.MinMovesToComplete,
                InitialMinCost = src.InitialMinCost,
                DifficultyPosition = src.DifficultyPosition,
                PressureValues = src.PressureValues != null ? new List<int>(src.PressureValues) : new List<int>(),
                PressureValueMean = src.PressureValueMean,
                PressureValueMin = src.PressureValueMin,
                PressureValueMax = src.PressureValueMax,
                PressureValueStdDev = src.PressureValueStdDev,
                DifficultyScore = src.DifficultyScore,
                FinalDifficulty = src.FinalDifficulty,
                ConsecutiveLowPressureCount = src.ConsecutiveLowPressureCount,
                TotalEarlyLowPressureCount = src.TotalEarlyLowPressureCount,
                ErrorMessage = src.ErrorMessage
            };
        }

        /// <summary>
        /// 加载关卡数据 - 增加缓存机制优化
        /// </summary>
        private static LevelData LoadLevelData(string levelName)
        {
            try
            {
                int levelId = 0;
                int.TryParse(levelName, out levelId);

                // 检查缓存(并发安全)
                if (levelId > 0)
                {
                    lock (_levelDataCacheLock)
                    {
                        if (_levelDataCache.TryGetValue(levelId, out LevelData cachedLevel))
                        {
                            return cachedLevel;
                        }
                    }
                }

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

                // 缓存结果(并发安全)
                if (levelId > 0 && levelData != null)
                {
                    lock (_levelDataCacheLock)
                    {
                        _levelDataCache[levelId] = levelData;
                    }
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
                var levelDatabase = UnityEditor.AssetDatabase.LoadAssetAtPath<LevelDatabase>(
                    "Assets/ArtRes/TMRes/StaticSettings/LevelDatabase.asset");
#else
                var levelDatabase = UnityEngine.Resources.Load<LevelDatabase>("LevelDatabase");
#endif

                if (levelDatabase == null)
                {
                    Debug.LogWarning("[BattleAnalyzer] 无法加载LevelDatabase,使用备用花色池");
                    return GetFallbackColorPool();
                }

                var allColors = levelDatabase.BaseThemeTiles
                    .Where(tile => tile != null && tile.ElementValue > 0)
                    .Select(tile => tile.ElementValue)
                    .Distinct()
                    .OrderBy(x => x)
                    .ToList();

                if (allColors.Count == 0)
                {
                    Debug.LogWarning("[BattleAnalyzer] LevelDatabase中无有效瓦片,使用备用花色池");
                    return GetFallbackColorPool();
                }

                Debug.Log($"[BattleAnalyzer] 从LevelDatabase加载花色池成功,共{allColors.Count}种花色");
                return allColors;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[BattleAnalyzer] 加载LevelDatabase失败: {ex.Message},使用备用花色池");
                return GetFallbackColorPool();
            }
        }

        /// <summary>
        /// 备用花色池 - 仅在无法访问LevelDatabase时使用
        /// </summary>
        private static List<int> GetFallbackColorPool()
        {
            var fallbackColors = new List<int>();

            // 9个色系,每个色系20种花色,总计180种花色
            int[] colorRanges = { 100, 200, 300, 400, 500, 600, 700, 800, 900 };
            foreach (int rangeStart in colorRanges)
            {
                for (int i = 1; i <= 20; i++)
                {
                    fallbackColors.Add(rangeStart + i);
                }
            }

            Debug.LogWarning($"[BattleAnalyzer] 使用备用花色池,共{fallbackColors.Count}种预定义花色");
            return fallbackColors;
        }

        /// <summary>
        /// 获取花色池大小
        /// </summary>
        private static int GetColorPoolSize()
        {
            if (_colorPoolCache != null)
            {
                return _colorPoolCache.Count;
            }

            _colorPoolCache = GetAvailableColorsFromDatabase();
            return _colorPoolCache.Count;
        }

        /// <summary>
        /// 创建可用花色列表
        /// </summary>
        private static List<int> CreateAvailableColorsDeterministic(int colorCount, int seed, int terrainId)
        {
            // 准备完整花色池(只读)
            if (_colorPoolCache == null)
            {
                _colorPoolCache = GetAvailableColorsFromDatabase();
            }
            var fullColorPool = _colorPoolCache;

            // 检查花色数量是否合法
            if (colorCount > fullColorPool.Count)
            {
                Debug.LogError($"[BattleAnalyzer] 请求的花色数量({colorCount})超过花色池大小({fullColorPool.Count})!");
                return new List<int>();
            }

            // 使用System.Random进行确定性Fisher-Yates洗牌
            // 将seed与地形/花色参数混合,保证不同关卡/花色数量下的分布稳定
            unchecked
            {
                int mixedSeed = seed ^ (terrainId * 73856093) ^ (colorCount * 19349663);
                var rng = new System.Random(mixedSeed);
                var shuffled = new List<int>(fullColorPool);
                for (int i = 0; i < shuffled.Count; i++)
                {
                    int j = i + rng.Next(shuffled.Count - i);
                    (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
                }
                return shuffled.GetRange(0, colorCount);
            }
        }

        /// <summary>
        /// 计算关卡总瓦片数量
        /// </summary>
        private static int CalculateTotalTileCount(LevelData levelData)
        {
            return levelData.Layers.Sum(layer => layer.tiles.Length);
        }

        // ========== 批量分析 ==========

        /// <summary>
        /// 批量运行分析
        /// </summary>
        public static List<AnalysisResult> RunBatchAnalysis(RunConfig config)
        {
            AlgoLogger.InitializeMainThread();
            CsvConfigManager.LoadCsvConfigs();
            var results = new List<AnalysisResult>();

            int maxColorPoolSize = GetColorPoolSize();
            Debug.Log($"[BattleAnalyzer] 花色池大小: {maxColorPoolSize}种花色");
            Debug.Log($"开始批量分析: {config.GetConfigDescription()}");

            bool useMultiBatch = config.BatchFilters != null && config.BatchFilters.Count > 0;

            // 内存监控初始化
            long initialMemory = GC.GetTotalMemory(false) / 1024 / 1024;

            // 预计算总任务数
            int totalTasks = 0;
            var rowConfigs = new List<(int rowIndex, int terrainId, int[][] experienceModes, int[] colorCounts)>();

            if (_csvConfigs == null || _csvConfigs.Count == 0)
            {
                Debug.LogError("CSV配置加载失败或为空!");
                return results;
            }

            Debug.Log($"CSV配置加载成功,共 {_csvConfigs.Count} 行配置");

            // 按行索引顺序处理
            int rowsToProcess = Math.Min(config.TestLevelCount, _csvConfigs.Count);
            for (int i = 0; i < rowsToProcess; i++)
            {
                var rowConfig = _csvConfigs[i];
                int rowIndex = rowConfig.RowIndex;
                int terrainId = rowConfig.TerrainId;

                var experienceModes = CsvConfigManager.ResolveExperienceModesByRow(config.ExperienceConfigEnum, rowIndex, config);
                var colorCounts = CsvConfigManager.ResolveColorCountsByRow(config.ColorCountConfigEnum, rowIndex);

                rowConfigs.Add((rowIndex, terrainId, experienceModes, colorCounts));

                if (config.IsFilteringEnabled)
                {
                    totalTasks += config.RequiredResultsPerTerrain;
                }
                else
                {
                    totalTasks += experienceModes.Length * colorCounts.Length * config.MaxSeedAttemptsPerConfig;
                }
            }

            Debug.Log($"处理行数: {rowConfigs.Count}, 总任务数: {totalTasks}");

            results.Capacity = totalTasks;
            int completedTasks = 0;
            object resultsLock = new object();

            // 预热: 加载行对应的关卡数据,避免工作线程首次IO
            foreach (var (rowIndex, terrainId, _, _) in rowConfigs)
            {
                LoadLevelData(terrainId.ToString());
            }

            Action<(int rowIndex, int terrainId, int[][] experienceModes, int[] colorCounts)> processRow = tuple =>
            {
                var (rowIndex, terrainId, experienceModes, colorCounts) = tuple;
                string levelName = terrainId.ToString();
                int configAttempts = 0;
                bool rowCompleted = false;

                // 生成配置组合列表
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
                    var shuffledConfigs = new List<(int[] expMode, int colorCount)>(allConfigCombinations);
                    var random = new System.Random(terrainId * 1000 + DateTime.Now.Millisecond);
                    for (int i = shuffledConfigs.Count - 1; i > 0; i--)
                    {
                        int j = random.Next(i + 1);
                        (shuffledConfigs[i], shuffledConfigs[j]) = (shuffledConfigs[j], shuffledConfigs[i]);
                    }
                    configSequence = shuffledConfigs;
                }
                else
                {
                    configSequence = allConfigCombinations;
                }

                if (useMultiBatch)
                {
                    var batchFoundCount = new Dictionary<string, int>();
                    var batchCompleted = new Dictionary<string, bool>();
                    foreach (var bf in config.BatchFilters)
                    {
                        var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                        batchFoundCount[name] = 0;
                        batchCompleted[name] = false;
                    }
                    Func<bool> AllBatchesDone = () => batchCompleted.Values.All(v => v);

                    foreach (var (experienceMode, colorCount) in configSequence)
                    {
                        if (AllBatchesDone()) break;
                        if (configAttempts >= config.MaxConfigAttemptsPerTerrain) break;

                        if (colorCount > maxColorPoolSize)
                        {
                            Debug.Log($"跳过配置 (行{rowIndex}|地形{terrainId}): 花色数量({colorCount})超过花色池上限({maxColorPoolSize})");
                            continue;
                        }
                        configAttempts++;
                        int seedAttempts = 0;

                        if (config.UseAverageFiltering)
                        {
                            var currentSeedsResults = new List<AnalysisResult>();
                            for (int i = 0; i < config.MaxSeedAttemptsPerConfig; i++)
                            {
                                int randomSeed = config.GetSeedForAttempt(terrainId, i);
                                seedAttempts++;
                                Interlocked.Increment(ref completedTasks);
#if !VERBOSE_ANALYZER_LOGGING
                                int ct = Volatile.Read(ref completedTasks);
                                if (ct % 500 == 0 || ct == 1)
                                {
                                    Debug.Log($"[进度] {ct}/{totalTasks} ({100f * ct / totalTasks:F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                                }
#endif
                                var raw = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                                raw.RowIndex = rowIndex;
                                raw.TerrainId = terrainId;
                                raw.UniqueId = null; // 将在克隆时分配
                                currentSeedsResults.Add(raw);
                            }
                            var aggregated = AggregateResultsByConfig(currentSeedsResults);
                            var agg = aggregated.Count > 0 ? aggregated[0] : null;
                            if (agg != null)
                            {
                                foreach (var bf in config.BatchFilters)
                                {
                                    var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                                    if (batchCompleted[name]) continue;
                                    if (bf.Matches(agg))
                                    {
                                        // 添加所有种子结果
                                        foreach (var r in currentSeedsResults)
                                        {
                                            var uid = $"BA_{Interlocked.Increment(ref _globalUniqueIdCounter):D6}";
                                            var copy = CloneForBatch(r, name, uid);
                                            lock (resultsLock) results.Add(copy);
                                        }

                                        // ✅ 修复: 计数应该按配置组计数，而不是按种子数量计数
                                        // 一个配置包含多个种子，但只算作1个找到的结果
                                        batchFoundCount[name]++;

                                        if (batchFoundCount[name] >= config.RequiredResultsPerTerrain)
                                            batchCompleted[name] = true;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // 优化的空运行计数逻辑：
                            // - 每个批次独立计数连续未命中次数
                            // - 只有当所有未完成的批次都达到空运行上限时才退出配置
                            // - 这样可以避免某批次提前退出而错过后续种子中的有效结果
                            var consecutiveMissesByBatch = new Dictionary<string, int>();
                            foreach (var bf in config.BatchFilters)
                            {
                                var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                                consecutiveMissesByBatch[name] = 0;
                            }

                            // 检查是否所有未完成的批次都已达到空运行上限
                            Func<bool> AllActiveBatchesExhausted = () =>
                            {
                                foreach (var bf in config.BatchFilters)
                                {
                                    var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                                    // 如果批次未完成且未达到空运行上限，则还有活跃批次
                                    if (!batchCompleted[name] &&
                                        consecutiveMissesByBatch[name] < config.MaxEmptySeedAttemptsPerConfig)
                                    {
                                        return false;
                                    }
                                }
                                return true; // 所有未完成批次都已空运行达上限
                            };

                            while (seedAttempts < config.MaxSeedAttemptsPerConfig)
                            {
                                if (AllBatchesDone()) break;
                                if (config.EnableGlobalFiltering && AllActiveBatchesExhausted()) break;

                                int randomSeed = config.GetSeedForAttempt(terrainId, seedAttempts);
                                seedAttempts++;
                                Interlocked.Increment(ref completedTasks);
#if !VERBOSE_ANALYZER_LOGGING
                                int ct = Volatile.Read(ref completedTasks);
                                if (ct % 500 == 0 || ct == 1)
                                {
                                    Debug.Log($"[进度] {ct}/{totalTasks} ({100f * ct / totalTasks:F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                                }
#endif
                                var raw = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                                raw.RowIndex = rowIndex;
                                raw.TerrainId = terrainId;
                                raw.UniqueId = null;

                                bool anyBatchMatched = false; // 标记本次种子是否有任何批次命中

                                foreach (var bf in config.BatchFilters)
                                {
                                    var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                                    if (batchCompleted[name]) continue;

                                    // 未开启平均值筛选时，只有成功通关的关卡才进入筛选
                                    if (bf.Matches(raw) && raw.GameCompleted)
                                    {
                                        var uid = $"BA_{Interlocked.Increment(ref _globalUniqueIdCounter):D6}";
                                        var copy = CloneForBatch(raw, name, uid);
                                        lock (resultsLock) results.Add(copy);
                                        batchFoundCount[name]++;
                                        consecutiveMissesByBatch[name] = 0; // 重置该批次的连续未命中计数
                                        anyBatchMatched = true;

                                        if (batchFoundCount[name] >= config.RequiredResultsPerTerrain)
                                        {
                                            batchCompleted[name] = true;
                                        }
                                    }
                                }

                                // 对所有未完成且未命中的批次增加连续未命中计数
                                if (config.EnableGlobalFiltering && !anyBatchMatched)
                                {
                                    foreach (var bf in config.BatchFilters)
                                    {
                                        var name = string.IsNullOrEmpty(bf.BatchName) ? "Batch" : bf.BatchName;
                                        if (!batchCompleted[name])
                                        {
                                            consecutiveMissesByBatch[name]++;
                                        }
                                    }
                                }

                                int ct2 = Volatile.Read(ref completedTasks);
                                if (ct2 % 200 == 0)
                                {
                                    GC.Collect(1, GCCollectionMode.Forced);
                                    GC.WaitForPendingFinalizers();
                                }
                            }
                        }
                    }
                }
                else
                {
                    foreach (var (experienceMode, colorCount) in configSequence)
                    {
                        if (rowCompleted) break;
                        if (configAttempts >= config.MaxConfigAttemptsPerTerrain) break;
                        if (colorCount > maxColorPoolSize)
                        {
                            Debug.Log($"跳过配置 (行{rowIndex}|地形{terrainId}): 花色数量({colorCount})超过花色池上限({maxColorPoolSize})");
                            continue;
                        }
                        configAttempts++;
                        var currentConfigResults = new List<AnalysisResult>();
                        int seedAttempts = 0;

                        if (config.IsFilteringEnabled && config.UseAverageFiltering)
                        {
                            for (int i = 0; i < config.MaxSeedAttemptsPerConfig; i++)
                            {
                                int randomSeed = config.GetSeedForAttempt(terrainId, i);
                                seedAttempts++;
                                Interlocked.Increment(ref completedTasks);
#if !VERBOSE_ANALYZER_LOGGING
                                int ct = Volatile.Read(ref completedTasks);
                                if (ct % 500 == 0 || ct == 1)
                                {
                                    Debug.Log($"[进度] {ct}/{totalTasks} ({100f * ct / totalTasks:F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                                }
#endif
                                var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                                result.RowIndex = rowIndex;
                                result.TerrainId = terrainId;
                                result.UniqueId = $"BA_{Interlocked.Increment(ref _globalUniqueIdCounter):D6}";
                                currentConfigResults.Add(result);
                            }
                            var aggregatedResults = AggregateResultsByConfig(currentConfigResults);
                            if (aggregatedResults.Count > 0 && config.MatchesCriteria(aggregatedResults[0]))
                            {
                                lock (resultsLock) results.AddRange(currentConfigResults);
                                rowCompleted = true;
                            }
                        }
                        else
                        {
                            while (seedAttempts < config.MaxSeedAttemptsPerConfig)
                            {
                                if (config.IsFilteringEnabled &&
                                    seedAttempts >= config.MaxEmptySeedAttemptsPerConfig &&
                                    currentConfigResults.Count == 0)
                                {
                                    break;
                                }
                                int randomSeed = config.GetSeedForAttempt(terrainId, seedAttempts);
                                seedAttempts++;
                                Interlocked.Increment(ref completedTasks);
#if !VERBOSE_ANALYZER_LOGGING
                                int ct = Volatile.Read(ref completedTasks);
                                if (ct % 500 == 0 || ct == 1)
                                {
                                    Debug.Log($"[进度] {ct}/{totalTasks} ({100f * ct / totalTasks:F1}%) - 当前: 行{rowIndex}|地形{terrainId}");
                                }
#endif
                                var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                                result.RowIndex = rowIndex;
                                result.TerrainId = terrainId;
                                result.UniqueId = $"BA_{Interlocked.Increment(ref _globalUniqueIdCounter):D6}";

                                int ct2 = Volatile.Read(ref completedTasks);
                                if (ct2 % 200 == 0)
                                {
                                    GC.Collect(1, GCCollectionMode.Forced);
                                    GC.WaitForPendingFinalizers();
                                }

                                if (config.IsFilteringEnabled)
                                {
                                    // 只有成功通关的关卡才进入筛选
                                    if (config.MatchesCriteria(result) && result.GameCompleted)
                                    {
                                        currentConfigResults.Add(result);
                                        if (currentConfigResults.Count >= config.RequiredResultsPerTerrain)
                                        {
                                            rowCompleted = true;
                                            break;
                                        }
                                    }
                                }
                                else
                                {
                                    currentConfigResults.Add(result);
                                }
                            }

                            if (config.IsFilteringEnabled)
                            {
                                if (rowCompleted)
                                {
                                    lock (resultsLock) results.AddRange(currentConfigResults);
                                    break;
                                }
                            }
                            else
                            {
                                lock (resultsLock) results.AddRange(currentConfigResults);
                            }
                        }
                    }
                }
            };

            if (config.EnableParallel)
            {
                Debug.Log($"并行执行: 行级并行, 最大并发={config.MaxParallelRows}");
                Parallel.ForEach(rowConfigs,
                    new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, config.MaxParallelRows) },
                    processRow);
            }
            else
            {
                foreach (var row in rowConfigs)
                {
                    processRow(row);
                }
            }

            long finalMemory = GC.GetTotalMemory(false) / 1024 / 1024;
            Debug.Log($"[内存监控] 最终内存: {finalMemory}MB, 总增长: {finalMemory - initialMemory}MB");
            Debug.Log($"批量分析完成: {results.Count} 个任务结果");

            return results;
        }

        /// <summary>
        /// 将结果按配置聚合,计算每个配置的平均值
        /// 注意: 传入的results已经过筛选,直接对所有结果计算平均值即可
        /// </summary>
        public static List<AggregatedResult> AggregateResultsByConfig(List<AnalysisResult> results)
        {
            // 分两层分组: 先按配置(ConfigKey), 再按批次(Batch)
            var groupedByConfig = new Dictionary<ConfigKey, Dictionary<string, List<AnalysisResult>>>();

            foreach (var r in results)
            {
                var key = new ConfigKey
                {
                    TerrainId = r.TerrainId,
                    ExperienceModeStr = $"[{string.Join(",", r.ExperienceMode)}]",
                    ColorCount = r.ColorCount
                };
                var batch = r.BatchName ?? string.Empty;

                if (!groupedByConfig.TryGetValue(key, out var byBatch))
                {
                    byBatch = new Dictionary<string, List<AnalysisResult>>();
                    groupedByConfig[key] = byBatch;
                }

                if (!byBatch.TryGetValue(batch, out var list))
                {
                    list = new List<AnalysisResult>();
                    byBatch[batch] = list;
                }

                list.Add(r);
            }

            var aggregatedResults = new List<AggregatedResult>();

            foreach (var kv in groupedByConfig)
            {
                foreach (var batchKv in kv.Value)
                {
                    var configResults = batchKv.Value;
                    // 安全去重: 同一批次中相同随机种子理论上只出现一次,仍按RandomSeed去重以防重复
                    var distinctResults = configResults
                        .GroupBy(r => r.RandomSeed)
                        .Select(g => g.First())
                        .ToList();

                    if (distinctResults.Count == 0) continue;

                    var firstResult = distinctResults.First();

                    // 统计成功和失败数量
                    var successfulResults = distinctResults.Where(r => r.GameCompleted).ToList();

                    var aggregated = new AggregatedResult
                    {
                        TerrainId = firstResult.TerrainId,
                        LevelName = firstResult.LevelName,
                        BatchName = firstResult.BatchName ?? string.Empty,
                        ExperienceMode = firstResult.ExperienceMode,
                        ColorCount = firstResult.ColorCount,
                        TotalTiles = firstResult.TotalTiles,
                        AlgorithmName = firstResult.AlgorithmName,
                        SeedCount = distinctResults.Count,
                        SeedList = distinctResults.Select(r => r.RandomSeed).ToList(),
                        WinRate = (double)successfulResults.Count / distinctResults.Count,

                        // 对所有传入的结果计算平均值(已经过筛选)
                        AvgPeakDockCount = distinctResults.Average(r => r.PeakDockCount),
                        AvgInitialMinCost = distinctResults.Average(r => r.InitialMinCost),
                        AvgPressureValueMean = distinctResults.Average(r => r.PressureValueMean),
                        AvgPressureValueMin = distinctResults.Average(r => r.PressureValueMin),
                        AvgPressureValueMax = distinctResults.Average(r => r.PressureValueMax),
                        AvgPressureValueStdDev = distinctResults.Average(r => r.PressureValueStdDev),
                        AvgDifficultyScore = distinctResults.Average(r => r.DifficultyScore),
                        AvgFinalDifficulty = distinctResults.Average(r => r.FinalDifficulty),
                        AvgConsecutiveLowPressureCount = distinctResults.Average(r => r.ConsecutiveLowPressureCount),
                        AvgTotalEarlyLowPressureCount = distinctResults.Average(r => r.TotalEarlyLowPressureCount),
                        AvgDifficultyPosition = distinctResults.Average(r => r.DifficultyPosition)
                    };

                    aggregatedResults.Add(aggregated);
                }
            }

            return aggregatedResults;
        }

        // ========== 导出方法 ==========

        /// <summary>
        /// 导出结果为CSV
        /// </summary>
        public static void ExportToCsv(List<AnalysisResult> results, string outputPath)
        {
            try
            {
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                using (var writer = new StreamWriter(outputPath, false, Encoding.UTF8, FILE_BUFFER_SIZE))
                {
                    // 写入CSV表头
                    writer.WriteLine("UniqueId,Batch,TerrainId,LevelName,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,RandomSeed," +
                                    "GameCompleted,GameDurationMs,CompletionStatus,InitialMinCost," +
                                    "DifficultyPosition,PeakDockCount,PressureValues," +
                                    "PressureValueMean,PressureValueMin,PressureValueMax,PressureValueStdDev,DifficultyScore,FinalDifficulty," +
                                    "ConsecutiveLowPressureCount,TotalEarlyLowPressureCount,ErrorMessage");

                    // 逐行写入数据
                    foreach (var result in results)
                    {
                        string expMode = $"[{string.Join(",", result.ExperienceMode)}]";
                        string pressureValues = result.PressureValues.Count > 1 ? string.Join(",", result.PressureValues.Take(result.PressureValues.Count - 1)) : "";

                        writer.WriteLine($"{result.UniqueId},\"{(result.BatchName ?? "")}\",{result.TerrainId},{result.LevelName},{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.RandomSeed}," +
                                       $"{result.GameCompleted},{result.GameDurationMs},\"{result.CompletionStatus}\",{result.InitialMinCost}," +
                                       $"{result.DifficultyPosition:F4},{result.PeakDockCount},\"{pressureValues}\"," +
                                       $"{result.PressureValueMean:F4},{result.PressureValueMin},{result.PressureValueMax},{result.PressureValueStdDev:F4},{result.DifficultyScore:F2},{result.FinalDifficulty}," +
                                       $"{result.ConsecutiveLowPressureCount},{result.TotalEarlyLowPressureCount},\"{result.ErrorMessage ?? ""}\"");
                    }
                }

                Debug.Log($"结果已导出到: {outputPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出CSV失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 导出聚合结果为CSV
        /// </summary>
        public static void ExportAggregatedToCsv(List<AggregatedResult> aggregatedResults, string outputPath)
        {
            try
            {
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                using (var writer = new StreamWriter(outputPath, false, Encoding.UTF8, FILE_BUFFER_SIZE))
                {
                    // 写入CSV表头
                    writer.WriteLine("TerrainId,LevelName,Batch,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,SeedCount,Seeds,WinRate," +
                                    "AvgPeakDockCount,AvgInitialMinCost," +
                                    "AvgPressureValueMean,AvgPressureValueMin,AvgPressureValueMax,AvgPressureValueStdDev," +
                                    "AvgDifficultyScore,AvgFinalDifficulty," +
                                    "AvgConsecutiveLowPressureCount,AvgTotalEarlyLowPressureCount," +
                                    "AvgDifficultyPosition");

                    // 逐行写入数据
                    foreach (var result in aggregatedResults)
                    {
                        string expMode = $"[{string.Join(",", result.ExperienceMode)}]";
                        string seeds = $"[{string.Join(",", result.SeedList)}]";

                        writer.WriteLine($"{result.TerrainId},{result.LevelName},\"{(result.BatchName ?? "")}\",{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.SeedCount},\"{seeds}\",{result.WinRate:F4}," +
                                       $"{result.AvgPeakDockCount:F2},{result.AvgInitialMinCost:F2}," +
                                       $"{result.AvgPressureValueMean:F4},{result.AvgPressureValueMin:F2},{result.AvgPressureValueMax:F2},{result.AvgPressureValueStdDev:F4}," +
                                       $"{result.AvgDifficultyScore:F2},{result.AvgFinalDifficulty:F2}," +
                                       $"{result.AvgConsecutiveLowPressureCount:F2},{result.AvgTotalEarlyLowPressureCount:F2}," +
                                       $"{result.AvgDifficultyPosition:F4}");
                    }
                }

                Debug.Log($"聚合结果已导出到: {outputPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出聚合CSV失败: {ex.Message}");
            }
        }

#if UNITY_EDITOR
        // ========== Unity Editor菜单 ==========

        /// <summary>
        /// Unity Editor菜单:运行BattleAnalyzer批量分析(弹出配置窗口)
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/运行批量分析")]
        public static void RunBatchAnalysisFromMenu()
        {
            BattleAnalyzerConfigWindow.ShowWindow();
        }

        /// <summary>
        /// Unity Editor菜单:快速运行批量分析(使用默认配置)
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/快速运行批量分析(默认配置)")]
        public static void RunBatchAnalysisFromMenuQuick()
        {
            var config = new RunConfig();
            config.OutputDirectory = Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults");

            Debug.Log($"=== 开始批量分析(默认配置) ===");

            var results = RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? "_FixedSeeds" : "_Random";
            string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

            ExportToCsv(results, csvPath);

            if (config.OutputPerConfigAverage)
            {
                var aggregatedResults = AggregateResultsByConfig(results);
                var aggregatedCsvPath = csvPath.Replace(".csv", "_Aggregated.csv");
                ExportAggregatedToCsv(aggregatedResults, aggregatedCsvPath);
            }

            Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
        }

        /// <summary>
        /// BattleAnalyzer配置窗口
        /// </summary>
        public class BattleAnalyzerConfigWindow : EditorWindow
        {
            // EditorPrefs存储键名常量
            private const string PREFS_PREFIX = "BattleAnalyzer_";
            private const string KEY_OUTPUT_DIR = PREFS_PREFIX + "OutputDirectory";
            private const string KEY_EXP_CONFIG = PREFS_PREFIX + "ExperienceConfigEnum";
            private const string KEY_COLOR_CONFIG = PREFS_PREFIX + "ColorCountConfigEnum";
            private const string KEY_TEST_LEVEL_COUNT = PREFS_PREFIX + "TestLevelCount";
            private const string KEY_ARRAY_LENGTH = PREFS_PREFIX + "ArrayLength";
            private const string KEY_MIN_VALUE = PREFS_PREFIX + "MinValue";
            private const string KEY_MAX_VALUE = PREFS_PREFIX + "MaxValue";
            private const string KEY_USE_RANDOM_CONFIG = PREFS_PREFIX + "UseRandomConfigSelection";
            private const string KEY_USE_FIXED_SEED = PREFS_PREFIX + "UseFixedSeed";
            private const string KEY_FIXED_SEED_VALUES = PREFS_PREFIX + "FixedSeedValues";
            private const string KEY_MAX_SEED_ATTEMPTS = PREFS_PREFIX + "MaxSeedAttemptsPerConfig";
            private const string KEY_MAX_EMPTY_ATTEMPTS = PREFS_PREFIX + "MaxEmptySeedAttemptsPerConfig";
            private const string KEY_OUTPUT_PER_CONFIG_AVG = PREFS_PREFIX + "OutputPerConfigAverage";
            private const string KEY_ENABLE_PARALLEL = PREFS_PREFIX + "EnableParallel";
            private const string KEY_MAX_PARALLEL_ROWS = PREFS_PREFIX + "MaxParallelRows";
            private const string KEY_ENABLE_GLOBAL_FILTERING = PREFS_PREFIX + "EnableGlobalFiltering";
            private const string KEY_USE_AVERAGE_FILTERING = PREFS_PREFIX + "UseAverageFiltering";
            private const string KEY_REQUIRED_RESULTS = PREFS_PREFIX + "RequiredResultsPerTerrain";
            private const string KEY_MAX_CONFIG_ATTEMPTS = PREFS_PREFIX + "MaxConfigAttemptsPerTerrain";

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
                config = new RunConfig();

                // 从EditorPrefs加载配置
                LoadConfigFromPrefs();

                // 确保输出目录有默认值
                if (string.IsNullOrEmpty(config.OutputDirectory))
                {
                    config.OutputDirectory = Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults");
                }

                // 加载种子值字符串
                if (config.FixedSeedValues != null && config.FixedSeedValues.Length > 0)
                {
                    seedValuesString = string.Join(",", config.FixedSeedValues);
                }

                // 默认添加一个批次(沿用全局筛选参数), 方便直接开始多批配置
                if (config.BatchFilters == null) config.BatchFilters = new List<RunConfig.BatchFilter>();
                if (config.BatchFilters.Count == 0)
                {
                    config.BatchFilters.Add(new RunConfig.BatchFilter
                    {
                        BatchName = "Batch-1",
                        DifficultyPositionRangeMin = config.GlobalDifficultyPositionRangeMin,
                        DifficultyPositionRangeMax = config.GlobalDifficultyPositionRangeMax,
                        DifficultyScoreRangeMin = config.GlobalDifficultyScoreRangeMin,
                        DifficultyScoreRangeMax = config.GlobalDifficultyScoreRangeMax,
                        ConsecutiveLowPressureRangeMin = config.GlobalConsecutiveLowPressureRangeMin,
                        ConsecutiveLowPressureRangeMax = config.GlobalConsecutiveLowPressureRangeMax,
                        TotalEarlyLowPressureRangeMin = config.GlobalTotalEarlyLowPressureRangeMin,
                        TotalEarlyLowPressureRangeMax = config.GlobalTotalEarlyLowPressureRangeMax
                    });
                }
            }

            /// <summary>
            /// 从EditorPrefs加载配置
            /// </summary>
            private void LoadConfigFromPrefs()
            {
                config.OutputDirectory = EditorPrefs.GetString(KEY_OUTPUT_DIR, Path.Combine(Application.dataPath, "验证器/Editor/BattleAnalysisResults"));
                config.ExperienceConfigEnum = EditorPrefs.GetInt(KEY_EXP_CONFIG, 1);
                config.ColorCountConfigEnum = EditorPrefs.GetInt(KEY_COLOR_CONFIG, 1);
                config.TestLevelCount = EditorPrefs.GetInt(KEY_TEST_LEVEL_COUNT, 5);
                config.ArrayLength = EditorPrefs.GetInt(KEY_ARRAY_LENGTH, 3);
                config.MinValue = EditorPrefs.GetInt(KEY_MIN_VALUE, 1);
                config.MaxValue = EditorPrefs.GetInt(KEY_MAX_VALUE, 9);
                config.UseRandomConfigSelection = EditorPrefs.GetBool(KEY_USE_RANDOM_CONFIG, false);
                config.UseFixedSeed = EditorPrefs.GetBool(KEY_USE_FIXED_SEED, false);

                string savedSeeds = EditorPrefs.GetString(KEY_FIXED_SEED_VALUES, "12345678,11111111,22222222,33333333,44444444,55555555,66666666,77777777,88888888,99999999");
                seedValuesString = savedSeeds;
                ParseSeedValues();

                config.MaxSeedAttemptsPerConfig = EditorPrefs.GetInt(KEY_MAX_SEED_ATTEMPTS, 1000);
                config.MaxEmptySeedAttemptsPerConfig = EditorPrefs.GetInt(KEY_MAX_EMPTY_ATTEMPTS, 100);
                config.OutputPerConfigAverage = EditorPrefs.GetBool(KEY_OUTPUT_PER_CONFIG_AVG, false);
                config.EnableParallel = EditorPrefs.GetBool(KEY_ENABLE_PARALLEL, false);
                config.MaxParallelRows = EditorPrefs.GetInt(KEY_MAX_PARALLEL_ROWS, 2);
                config.EnableGlobalFiltering = EditorPrefs.GetBool(KEY_ENABLE_GLOBAL_FILTERING, false);
                config.UseAverageFiltering = EditorPrefs.GetBool(KEY_USE_AVERAGE_FILTERING, false);
                config.RequiredResultsPerTerrain = EditorPrefs.GetInt(KEY_REQUIRED_RESULTS, 1);
                config.MaxConfigAttemptsPerTerrain = EditorPrefs.GetInt(KEY_MAX_CONFIG_ATTEMPTS, 1000);
            }

            /// <summary>
            /// 保存配置到EditorPrefs
            /// </summary>
            private void SaveConfigToPrefs()
            {
                EditorPrefs.SetString(KEY_OUTPUT_DIR, config.OutputDirectory);
                EditorPrefs.SetInt(KEY_EXP_CONFIG, config.ExperienceConfigEnum);
                EditorPrefs.SetInt(KEY_COLOR_CONFIG, config.ColorCountConfigEnum);
                EditorPrefs.SetInt(KEY_TEST_LEVEL_COUNT, config.TestLevelCount);
                EditorPrefs.SetInt(KEY_ARRAY_LENGTH, config.ArrayLength);
                EditorPrefs.SetInt(KEY_MIN_VALUE, config.MinValue);
                EditorPrefs.SetInt(KEY_MAX_VALUE, config.MaxValue);
                EditorPrefs.SetBool(KEY_USE_RANDOM_CONFIG, config.UseRandomConfigSelection);
                EditorPrefs.SetBool(KEY_USE_FIXED_SEED, config.UseFixedSeed);
                EditorPrefs.SetString(KEY_FIXED_SEED_VALUES, seedValuesString);
                EditorPrefs.SetInt(KEY_MAX_SEED_ATTEMPTS, config.MaxSeedAttemptsPerConfig);
                EditorPrefs.SetInt(KEY_MAX_EMPTY_ATTEMPTS, config.MaxEmptySeedAttemptsPerConfig);
                EditorPrefs.SetBool(KEY_OUTPUT_PER_CONFIG_AVG, config.OutputPerConfigAverage);
                EditorPrefs.SetBool(KEY_ENABLE_PARALLEL, config.EnableParallel);
                EditorPrefs.SetInt(KEY_MAX_PARALLEL_ROWS, config.MaxParallelRows);
                EditorPrefs.SetBool(KEY_ENABLE_GLOBAL_FILTERING, config.EnableGlobalFiltering);
                EditorPrefs.SetBool(KEY_USE_AVERAGE_FILTERING, config.UseAverageFiltering);
                EditorPrefs.SetInt(KEY_REQUIRED_RESULTS, config.RequiredResultsPerTerrain);
                EditorPrefs.SetInt(KEY_MAX_CONFIG_ATTEMPTS, config.MaxConfigAttemptsPerTerrain);
            }

            private void OnGUI()
            {
                scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);

                GUILayout.Label("BattleAnalyzer 批量分析配置", EditorStyles.boldLabel);
                GUILayout.Space(10);

                // CSV配置选择器
                GUILayout.Label("CSV配置选择器", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                config.ExperienceConfigEnum = EditorGUILayout.IntField(new GUIContent("体验模式枚举",
                    "1=exp-fix-1, -1=exp-range-1所有配置, -2=数组排列组合"), config.ExperienceConfigEnum);

                config.ColorCountConfigEnum = EditorGUILayout.IntField(new GUIContent("花色数量枚举",
                    "1=type-count-1, -1=type-range-1所有配置, -2=动态范围"), config.ColorCountConfigEnum);

                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 测试参数
                GUILayout.Label("测试参数", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                config.TestLevelCount = EditorGUILayout.IntField(new GUIContent("测试地形数量"), config.TestLevelCount);

                // 并行配置
                config.EnableParallel = EditorGUILayout.Toggle(new GUIContent("启用行级并行"), config.EnableParallel);
                if (config.EnableParallel)
                {
                    config.MaxParallelRows = EditorGUILayout.IntField(new GUIContent("最大并行行数"), Mathf.Max(1, config.MaxParallelRows));
                }

                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 排列组合配置
                GUILayout.Label("排列组合配置 (ExperienceConfigEnum = -2时生效)", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                config.ArrayLength = EditorGUILayout.IntField(new GUIContent("数组长度"), config.ArrayLength);
                config.MinValue = EditorGUILayout.IntField(new GUIContent("最小值"), config.MinValue);
                config.MaxValue = EditorGUILayout.IntField(new GUIContent("最大值"), config.MaxValue);

                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 配置选择策略
                GUILayout.Label("配置选择策略", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                config.UseRandomConfigSelection = EditorGUILayout.Toggle(new GUIContent("随机选择配置", "true=在范围内随机选择(不重复),false=按顺序遍历"), config.UseRandomConfigSelection);

                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 随机种子配置
                GUILayout.Label("随机种子配置", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                config.UseFixedSeed = EditorGUILayout.Toggle(new GUIContent("使用固定种子"), config.UseFixedSeed);

                EditorGUILayout.LabelField("固定种子值列表(逗号分隔):");
                seedValuesString = EditorGUILayout.TextArea(seedValuesString, GUILayout.Height(40));

                config.MaxSeedAttemptsPerConfig = EditorGUILayout.IntField(new GUIContent("每配置最大种子尝试次数"), config.MaxSeedAttemptsPerConfig);
                config.MaxEmptySeedAttemptsPerConfig = EditorGUILayout.IntField(new GUIContent("每配置最大空运行次数", "当配置连续x次都找不到符合条件的种子时提前退出"), config.MaxEmptySeedAttemptsPerConfig);

                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 批次筛选配置
                GUILayout.Label("批次筛选配置(同一地形可配置多批, 共用种子生成)", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");
                // 多批列表
                if (config.BatchFilters == null) config.BatchFilters = new List<RunConfig.BatchFilter>();
                for (int i = 0; i < config.BatchFilters.Count; i++)
                {
                    var bf = config.BatchFilters[i];
                    EditorGUILayout.BeginVertical("box");
                    EditorGUILayout.BeginHorizontal();
                    bf.BatchName = EditorGUILayout.TextField("批次名称", bf.BatchName);
                    if (GUILayout.Button("删除", GUILayout.Width(60)))
                    {
                        config.BatchFilters.RemoveAt(i);
                        i--;
                        EditorGUILayout.EndHorizontal();
                        EditorGUILayout.EndVertical();
                        continue;
                    }
                    EditorGUILayout.EndHorizontal();

                    EditorGUI.indentLevel++;
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("难点位置 (Pos)", GUILayout.Width(120));
                    bf.DifficultyPositionRangeMin = EditorGUILayout.FloatField(bf.DifficultyPositionRangeMin, GUILayout.Width(60));
                    EditorGUILayout.LabelField("~", GUILayout.Width(15));
                    bf.DifficultyPositionRangeMax = EditorGUILayout.FloatField(bf.DifficultyPositionRangeMax, GUILayout.Width(60));
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("难度分数 (Score)", GUILayout.Width(120));
                    bf.DifficultyScoreRangeMin = EditorGUILayout.FloatField(bf.DifficultyScoreRangeMin, GUILayout.Width(60));
                    EditorGUILayout.LabelField("~", GUILayout.Width(15));
                    bf.DifficultyScoreRangeMax = EditorGUILayout.FloatField(bf.DifficultyScoreRangeMax, GUILayout.Width(60));
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("连续低压力 (ConLP)", GUILayout.Width(120));
                    bf.ConsecutiveLowPressureRangeMin = EditorGUILayout.IntField(bf.ConsecutiveLowPressureRangeMin, GUILayout.Width(60));
                    EditorGUILayout.LabelField("~", GUILayout.Width(15));
                    bf.ConsecutiveLowPressureRangeMax = EditorGUILayout.IntField(bf.ConsecutiveLowPressureRangeMax, GUILayout.Width(60));
                    EditorGUILayout.EndHorizontal();

                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("前期低压力总数 (TotLP)", GUILayout.Width(120));
                    bf.TotalEarlyLowPressureRangeMin = EditorGUILayout.IntField(bf.TotalEarlyLowPressureRangeMin, GUILayout.Width(60));
                    EditorGUILayout.LabelField("~", GUILayout.Width(15));
                    bf.TotalEarlyLowPressureRangeMax = EditorGUILayout.IntField(bf.TotalEarlyLowPressureRangeMax, GUILayout.Width(60));
                    EditorGUILayout.EndHorizontal();
                    EditorGUI.indentLevel--;

                    EditorGUILayout.EndVertical();
                }

                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("添加批次"))
                {
                    config.BatchFilters.Add(new RunConfig.BatchFilter
                    {
                        BatchName = $"Batch-{config.BatchFilters.Count + 1}",
                        DifficultyPositionRangeMin = 0.5f,
                        DifficultyPositionRangeMax = 0.99f,
                        DifficultyScoreRangeMin = 150f,
                        DifficultyScoreRangeMax = 300f,
                        ConsecutiveLowPressureRangeMin = 0,
                        ConsecutiveLowPressureRangeMax = 100,
                        TotalEarlyLowPressureRangeMin = 4,
                        TotalEarlyLowPressureRangeMax = 7
                    });
                }
                if (GUILayout.Button("清空批次"))
                {
                    config.BatchFilters.Clear();
                }
                EditorGUILayout.EndHorizontal();

                // 共用项: 是否使用平均值筛选 + 终止条件(不分别配置)
                GUILayout.Space(4);
                config.UseAverageFiltering = EditorGUILayout.Toggle(new GUIContent("使用平均值筛选", "true=跑满种子后对平均值筛选,false=每个种子立即筛选"), config.UseAverageFiltering);
                config.RequiredResultsPerTerrain = EditorGUILayout.IntField(new GUIContent("每地形需要结果数量(每批)"), config.RequiredResultsPerTerrain);
                config.MaxConfigAttemptsPerTerrain = EditorGUILayout.IntField(new GUIContent("每地形最大尝试配置数量(共享)"), config.MaxConfigAttemptsPerTerrain);
                config.EnableGlobalFiltering = EditorGUILayout.Toggle(new GUIContent("仍使用旧的单批筛选开关(无批次时生效)"), config.EnableGlobalFiltering);
                EditorGUILayout.EndVertical();
                GUILayout.Space(10);

                // 输出配置
                GUILayout.Label("输出配置", EditorStyles.boldLabel);
                EditorGUILayout.BeginVertical("box");

                EditorGUILayout.LabelField("输出目录:");
                config.OutputDirectory = EditorGUILayout.TextField(config.OutputDirectory);

                if (GUILayout.Button("选择输出目录"))
                {
                    string selectedPath = EditorUtility.OpenFolderPanel("选择输出目录", config.OutputDirectory, "");
                    if (!string.IsNullOrEmpty(selectedPath))
                    {
                        config.OutputDirectory = selectedPath;
                    }
                }

                config.OutputPerConfigAverage = EditorGUILayout.Toggle(
                    new GUIContent("仅输出每配置平均值"),
                    config.OutputPerConfigAverage);

                EditorGUILayout.EndVertical();
                GUILayout.Space(20);

                // 执行按钮
                EditorGUILayout.BeginHorizontal();

                if (GUILayout.Button("运行批量分析", GUILayout.Height(40)))
                {
                    RunAnalysis();
                }

                if (GUILayout.Button("重置为默认配置", GUILayout.Height(40)))
                {
                    ResetToDefault();
                }

                EditorGUILayout.EndHorizontal();

                EditorGUILayout.EndScrollView();
            }

            private void RunAnalysis()
            {
                ParseSeedValues();

                // 保存配置到EditorPrefs
                SaveConfigToPrefs();

                if (!Directory.Exists(config.OutputDirectory))
                {
                    Directory.CreateDirectory(config.OutputDirectory);
                }

                Debug.Log($"=== 开始批量分析 ===");

                var results = BattleAnalyzerRunner.RunBatchAnalysis(config);

                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string seedSuffix = config.UseFixedSeed ? "_FixedSeeds" : "_Random";
                string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
                var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

                BattleAnalyzerRunner.ExportToCsv(results, csvPath);

                if (config.OutputPerConfigAverage)
                {
                    var aggregatedResults = BattleAnalyzerRunner.AggregateResultsByConfig(results);
                    var aggregatedCsvPath = csvPath.Replace(".csv", "_Aggregated.csv");
                    BattleAnalyzerRunner.ExportAggregatedToCsv(aggregatedResults, aggregatedCsvPath);
                }

                Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");

                Close();
            }

            private void ParseSeedValues()
            {
                if (string.IsNullOrWhiteSpace(seedValuesString))
                {
                    config.FixedSeedValues = new int[] { 12345678, 11111111, 22222222, 33333333, 44444444 };
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


