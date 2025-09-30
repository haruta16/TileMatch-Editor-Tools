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
        /// <summary>
        /// CSV配置行数据 - 统一使用BatchLevelEvaluatorSimple的数据结构
        /// </summary>
        public class CsvLevelConfig
        {
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
        }

        /// <summary>
        /// 自动游戏分析结果
        /// </summary>
        public class AnalysisResult
        {
            public string UniqueId { get; set; } // 唯一标识符
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
            public bool HasValidConfig => PositionRange.HasValue || ScoreRange.HasValue;

            /// <summary>
            /// 检查分析结果是否符合地形特定筛选条件
            /// </summary>
            public bool MatchesCriteria(AnalysisResult result)
            {
                bool positionMatch = true;
                bool scoreMatch = true;

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

                return positionMatch && scoreMatch;
            }

            /// <summary>
            /// 获取配置描述
            /// </summary>
            public string GetDescription()
            {
                var parts = new List<string>();
                if (PositionRange.HasValue)
                    parts.Add($"Position[{PositionRange.Value.min:F2}-{PositionRange.Value.max:F2}]");
                if (ScoreRange.HasValue)
                    parts.Add($"Score[{ScoreRange.Value.min:F0}-{ScoreRange.Value.max:F0}]");
                return parts.Count > 0 ? string.Join(", ", parts) : "无筛选条件";
            }
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
            public int TestLevelCount = 5; // 测试关卡数量

            [Header("=== 排列组合配置 (ExperienceConfigEnum = -2时生效) ===")]
            public int ArrayLength = 3; // 数组长度
            public int MinValue = 1; // 最小值
            public int MaxValue = 9; // 最大值

            [Header("=== 随机种子配置 ===")]
            public bool UseFixedSeed = true; // 是否使用固定种子：true=结果可重现，false=完全随机
            public int[] FixedSeedValues = { 12345678, 11111111, 22222222, 33333333, 44444444, 55555555, 66666666, 77777777, 88888888, 99999999 }; // 固定种子值列表
            public int RunsPerLevel = 5; // 每个关卡运行次数：用于生成多样化数据

            [Header("=== 输出配置 ===")]
            public string OutputDirectory = "BattleAnalysisResults";

            [Header("=== 筛选配置 ===")]
            public bool UseTerrainSpecificFiltering = true; // 是否使用地形特定筛选（从CSV读取）
            public bool EnableGlobalFiltering = false; // 是否启用全局筛选（作为fallback）
            public float GlobalDifficultyPositionRangeMin = 0.55f; // 全局难点位置范围最小值
            public float GlobalDifficultyPositionRangeMax = 0.8f; // 全局难点位置范围最大值
            public float GlobalDifficultyScoreRangeMin = 150f; // 全局难度分数范围最小值
            public float GlobalDifficultyScoreRangeMax = 300f; // 全局难度分数范围最大值
            public int RequiredResultsPerTerrain = 1; // 每个地形需要找到的符合条件结果数量

            [Header("=== 策略切换配置 ===")]
            public bool EnableStrategicSwitching = false; // 是否启用策略性配置切换
            public int MaxConfigAttemptsPerTerrain = 100; // 每个地形最大尝试配置数量

            /// <summary>
            /// 获取用于测试的随机种子
            /// </summary>
            public int GetSeedForRun(int levelIndex, int runIndex)
            {
                if (UseFixedSeed)
                {
                    if (FixedSeedValues != null && FixedSeedValues.Length > 0)
                    {
                        int seedIndex = runIndex % FixedSeedValues.Length;
                        return FixedSeedValues[seedIndex];
                    }
                    else
                    {
                        return 12345678 + levelIndex * 1000 + runIndex;
                    }
                }
                else
                {
                    return UnityEngine.Random.Range(1, int.MaxValue);
                }
            }

            /// <summary>
            /// 检查分析结果是否符合筛选条件（优先使用地形特定配置）
            /// </summary>
            public bool MatchesCriteria(AnalysisResult result)
            {
                // 优先使用地形特定筛选
                if (UseTerrainSpecificFiltering)
                {
                    var terrainConfig = CsvConfigManager.GetTerrainFilterConfig(result.TerrainId);
                    if (terrainConfig.HasValidConfig)
                    {
                        return terrainConfig.MatchesCriteria(result);
                    }
                }

                // Fallback到全局筛选
                if (EnableGlobalFiltering)
                {
                    return result.DifficultyPosition >= GlobalDifficultyPositionRangeMin &&
                           result.DifficultyPosition <= GlobalDifficultyPositionRangeMax &&
                           result.DifficultyScore >= GlobalDifficultyScoreRangeMin &&
                           result.DifficultyScore <= GlobalDifficultyScoreRangeMax;
                }

                // 如果都没有启用筛选，返回true
                return true;
            }

            /// <summary>
            /// 检查是否启用了任何筛选
            /// </summary>
            public bool IsFilteringEnabled => UseTerrainSpecificFiltering || EnableGlobalFiltering;

            /// <summary>
            /// 获取筛选条件描述
            /// </summary>
            public string GetFilterDescription()
            {
                if (!IsFilteringEnabled) return "筛选已禁用";

                var parts = new List<string>();
                if (UseTerrainSpecificFiltering)
                    parts.Add("地形特定筛选");
                if (EnableGlobalFiltering)
                    parts.Add($"全局筛选[Position:{GlobalDifficultyPositionRangeMin:F2}-{GlobalDifficultyPositionRangeMax:F2}, Score:{GlobalDifficultyScoreRangeMin:F0}-{GlobalDifficultyScoreRangeMax:F0}]");

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
                    -2 => "动态花色范围(总tile数/3的40%-80%)",
                    _ => $"配置{ColorCountConfigEnum}"
                };

                string seedMode = UseFixedSeed ? $"固定种子列表({FixedSeedValues?.Length ?? 0}个)" : "随机种子";
                string filterMode = IsFilteringEnabled ? $", 筛选[{GetFilterDescription()}]" : "";
                string strategyMode = EnableStrategicSwitching ? $", 策略切换[最多{MaxConfigAttemptsPerTerrain}次尝试]" : "";

                return $"体验模式[{expMode}], 花色数量[{colorMode}], {seedMode}, 每关卡{RunsPerLevel}次{filterMode}{strategyMode}";
            }
        }

        private static Dictionary<int, CsvLevelConfig> _csvConfigs = null;
        private static readonly object _csvLock = new object();
        private static Dictionary<int, LevelData> _levelDataCache = new Dictionary<int, LevelData>();
        private static readonly Dictionary<int, List<int>> _standardColorsCache = new Dictionary<int, List<int>>();

        /// <summary>
        /// CSV配置管理器 - 统一的配置加载和解析服务
        /// </summary>
        public static class CsvConfigManager
        {
            /// <summary>
            /// 加载CSV配置数据 - 线程安全优化版本
            /// </summary>
            public static void LoadCsvConfigs()
            {
                if (_csvConfigs != null) return;

                lock (_csvLock)
                {
                    if (_csvConfigs != null) return; // 双重检查锁定模式

                    _csvConfigs = new Dictionary<int, CsvLevelConfig>();

                    try
                    {
                        string csvPath = Path.Combine(Application.dataPath, "验证器/Editor/all_level.csv");
                        if (!File.Exists(csvPath))
                        {
                            Debug.LogError($"CSV配置文件不存在: {csvPath}");
                            return;
                        }

                        using (var fileStream = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.Read, 65536))
                        using (var reader = new StreamReader(fileStream, Encoding.UTF8, true, 65536))
                        {
                            reader.ReadLine(); // 跳过表头
                            string line;
                            while ((line = reader.ReadLine()) != null)
                            {
                                var parts = CsvParser.ParseCsvLine(line);
                                if (parts.Length >= 17 && int.TryParse(parts[0], out int terrainId))
                                {
                                    var config = new CsvLevelConfig
                                    {
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
                                        ScoreRange = CsvParser.ParseFloatRange(parts[16])
                                    };
                                    _csvConfigs[terrainId] = config;
                                }
                            }
                        }

                        Debug.Log($"成功加载CSV配置: {_csvConfigs.Count} 个地形");
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"加载CSV配置失败: {ex.Message}");
                        _csvConfigs = new Dictionary<int, CsvLevelConfig>();
                    }
                }
            }

            /// <summary>
            /// 根据枚举配置解析体验模式数组 - 支持-1全配置模式和-2排列组合模式
            /// </summary>
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
                        // 固定配置：使用特定地形的配置
                        if (!_csvConfigs.TryGetValue(terrainId, out var config))
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
            /// 根据枚举配置解析花色数量数组 - 支持-1全配置模式，-2动态范围模式
            /// </summary>
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
                        // 固定配置：使用特定地形的配置
                        if (!_csvConfigs.TryGetValue(terrainId, out var config))
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

                foreach (var row in _csvConfigs.Values)
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
            /// 生成基于总tile数的动态花色范围：总tile数/3的40%-80%，向下取整
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
                    int maxColorCount = Mathf.FloorToInt(totalGroups * 1.0f);

                    // 保证最小值至少为1
                    minColorCount = Math.Max(1, minColorCount);
                    maxColorCount = Math.Max(minColorCount, maxColorCount);

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

                var results = _csvConfigs.Values
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
            /// 获取地形特定的筛选配置
            /// </summary>
            public static TerrainFilterConfig GetTerrainFilterConfig(int terrainId)
            {
                LoadCsvConfigs();

                if (_csvConfigs.TryGetValue(terrainId, out var config))
                {
                    return new TerrainFilterConfig
                    {
                        TerrainId = terrainId,
                        PositionRange = config.PositionRange,
                        ScoreRange = config.ScoreRange
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
            private static readonly StringBuilder _reusableStringBuilder = new StringBuilder(256); // 复用StringBuilder

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

                Debug.Log($"=== 压力分析结果(均值) ===");
                Debug.Log($"有效运行数: {validResults.Count}/{runCount}, 通关成功率: {(float)completedCount/validResults.Count:P1}");
                Debug.Log($"成功关卡统计基数: {successfulResults.Count}个成功通关的关卡");
                Debug.Log($"使用的随机种子: [{string.Join(",", usedSeeds)}]");
                Debug.Log($"总步数(均值): {avgTotalMoves:F1}, 成功消除组数(均值): {avgSuccessfulGroups:F1}");
                Debug.Log($"峰值Dock数量(均值): {avgPeakDockCount:F1}, 开局最小Cost(均值): {avgInitialMinCost:F1}");
                Debug.Log($"压力统计(均值) - 均值: {avgPressureValueMean:F2}, 最小值: {avgPressureValueMin:F1}, 最大值: {avgPressureValueMax:F1}");
                Debug.Log($"压力标准差(均值): {avgPressureValueStdDev:F2}, 难度分数(均值): {avgDifficultyScore:F2}, 最终难度(均值): {avgFinalDifficulty:F1}/5");
                Debug.Log($"难点位置(均值): {avgDifficultyPosition:F2} (0=开局, 1=结尾) - 仅基于成功通关关卡");
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
        /// 创建可用花色列表 - 使用原始Fisher-Yates算法保持结果一致性
        /// </summary>
        private static List<int> CreateAvailableColors(int colorCount)
        {
            // 检查缓存
            if (_standardColorsCache.TryGetValue(colorCount, out List<int> cachedColors))
            {
                return new List<int>(cachedColors); // 返回副本防止修改
            }

            var standardColors = new int[] { 101, 102, 103, 201, 202, 301, 302, 401, 402, 403, 501, 502, 601, 602, 701, 702, 703, 801, 802 };

            List<int> result;
            if (colorCount <= standardColors.Length)
            {
                // 使用原始Fisher-Yates洗牌算法（正向遍历）保持结果一致性
                var shuffled = new List<int>(standardColors);
                for (int i = 0; i < shuffled.Count; i++)
                {
                    int randomIndex = UnityEngine.Random.Range(i, shuffled.Count);
                    (shuffled[i], shuffled[randomIndex]) = (shuffled[randomIndex], shuffled[i]);
                }
                result = shuffled.GetRange(0, colorCount);
            }
            else
            {
                result = new List<int>(standardColors);
            }

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

            // 获取标准花色池上限
            const int maxColorPoolSize = 19;

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

            // 预计算总任务数 - 从CSV配置中获取真实的terrainId
            int totalTasks = 0;
            var levelConfigs = new Dictionary<string, (int[][] experienceModes, int[] colorCounts)>();

            // 从CSV配置中获取所有terrainId，按数量限制
            if (_csvConfigs == null || _csvConfigs.Count == 0)
            {
                Debug.LogError("CSV配置加载失败或为空，无法进行批量分析！");
                return results;
            }

            // 按照CSV文件的行顺序（而不是数值大小）获取terrainId
            var availableTerrainIds = new List<int>();

            // 重新按CSV文件顺序读取
            try
            {
                string csvPath = Path.Combine(Application.dataPath, "验证器/Editor/all_level.csv");
                using (var reader = new StreamReader(csvPath, Encoding.UTF8))
                {
                    reader.ReadLine(); // 跳过表头
                    string line;
                    int count = 0;
                    while ((line = reader.ReadLine()) != null && count < config.TestLevelCount)
                    {
                        var parts = CsvParser.ParseCsvLine(line);
                        if (parts.Length >= 1 && int.TryParse(parts[0], out int terrainId))
                        {
                            if (_csvConfigs.ContainsKey(terrainId))
                            {
                                availableTerrainIds.Add(terrainId);
                                count++;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"按行顺序读取CSV失败，使用排序后的方式: {ex.Message}");
                availableTerrainIds = _csvConfigs.Keys.OrderBy(x => x).Take(config.TestLevelCount).ToList();
            }
            Debug.Log($"CSV配置加载成功，共 {_csvConfigs.Count} 个地形配置");

            foreach (int terrainId in availableTerrainIds)
            {
                string levelName = terrainId.ToString();
                var experienceModes = CsvConfigManager.ResolveExperienceModesWithConfig(config.ExperienceConfigEnum, terrainId, config);
                var colorCounts = CsvConfigManager.ResolveColorCounts(config.ColorCountConfigEnum, terrainId);
                levelConfigs[levelName] = (experienceModes, colorCounts);
                totalTasks += experienceModes.Length * colorCounts.Length * config.RunsPerLevel;
            }

            Debug.Log($"关卡数量: {availableTerrainIds.Count}, 总任务数: {totalTasks}");
            Debug.Log($"使用的TerrainId列表: [{string.Join(",", availableTerrainIds)}]");

            // 预分配结果列表容量优化
            results.Capacity = totalTasks;
            int completedTasks = 0;
            int skippedTasks = 0;
            int uniqueIdCounter = 1; // 唯一ID计数器

            foreach (var kvp in levelConfigs)
            {
                string levelName = kvp.Key;
                var (experienceModes, colorCounts) = kvp.Value;
                int terrainId = int.Parse(levelName); // 直接解析levelName为terrainId

                // 地形级别的策略性配置搜索
                var terrainValidResults = new List<AnalysisResult>(); // 当前地形的所有符合条件结果
                int configAttempts = 0;

                foreach (var experienceMode in experienceModes)
                {
                    // 检查是否已找到足够的符合条件结果
                    if (config.IsFilteringEnabled && terrainValidResults.Count >= config.RequiredResultsPerTerrain) break;
                    if (configAttempts >= config.MaxConfigAttemptsPerTerrain && config.EnableStrategicSwitching) break;

                    foreach (var colorCount in colorCounts)
                    {
                        // 检查是否已找到足够的符合条件结果
                        if (config.IsFilteringEnabled && terrainValidResults.Count >= config.RequiredResultsPerTerrain) break;

                        // 检查花色数量是否超过花色池上限
                        if (colorCount > maxColorPoolSize)
                        {
                            skippedTasks += config.RunsPerLevel;
                            Debug.Log($"跳过关卡 {terrainId}: 花色数量({colorCount})超过花色池上限({maxColorPoolSize})");
                            continue;
                        }

                        // 当前配置组合的所有种子遍历
                        var currentConfigResults = new List<AnalysisResult>();
                        bool currentConfigFoundValid = false;

                        foreach (var runIndex in Enumerable.Range(0, config.RunsPerLevel))
                        {
                            int randomSeed = config.GetSeedForRun(terrainId, runIndex);

                            completedTasks++;
                            if (config.IsFilteringEnabled)
                            {
                                var terrainConfig = CsvConfigManager.GetTerrainFilterConfig(terrainId);
                                string filterInfo = terrainConfig.HasValidConfig ? $"地形筛选[{terrainConfig.GetDescription()}]" : "全局筛选";
                                Debug.Log($"[{completedTasks}/{totalTasks - skippedTasks}] 搜索关卡 {terrainId} (配置尝试{configAttempts + 1}): " +
                                         $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}, {filterInfo}");
                            }
                            else
                            {
                                Debug.Log($"[{completedTasks}/{totalTasks - skippedTasks}] 分析关卡 {terrainId}: " +
                                         $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}");
                            }

                            var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                            result.TerrainId = terrainId;
                            result.UniqueId = $"BA_{uniqueIdCounter:D6}"; // 生成唯一ID：BA_000001, BA_000002...
                            uniqueIdCounter++;

                            // 根据筛选模式决定是否添加结果
                            if (config.IsFilteringEnabled)
                            {
                                // 筛选模式：只添加符合条件的结果
                                if (config.MatchesCriteria(result))
                                {
                                    Debug.Log($"  ✓ 找到符合条件的配置！DifficultyPosition={result.DifficultyPosition:F3}, DifficultyScore={result.DifficultyScore:F1} (地形{terrainId}第{terrainValidResults.Count + 1}个)");
                                    currentConfigResults.Add(result);
                                    terrainValidResults.Add(result);
                                    currentConfigFoundValid = true;
                                }
                            }
                            else
                            {
                                // 非筛选模式：添加所有结果
                                currentConfigResults.Add(result);
                            }
                        }

                        // 添加当前配置的筛选结果
                        results.AddRange(currentConfigResults);
                        configAttempts++;

                        // 如果当前配置找到了符合条件的结果，且已达到要求数量，跳出花色循环
                        if (config.IsFilteringEnabled && terrainValidResults.Count >= config.RequiredResultsPerTerrain) break;
                    }

                    // 如果找到了足够的符合条件的配置，且启用了筛选，跳出体验模式循环
                    if (config.IsFilteringEnabled && terrainValidResults.Count >= config.RequiredResultsPerTerrain) break;
                }

                if (config.IsFilteringEnabled)
                {
                    if (terrainValidResults.Count == 0)
                    {
                        Debug.LogWarning($"地形 {terrainId} 在 {configAttempts} 个配置尝试后未找到符合筛选条件的结果");
                    }
                    else if (terrainValidResults.Count < config.RequiredResultsPerTerrain)
                    {
                        Debug.LogWarning($"地形 {terrainId} 仅找到 {terrainValidResults.Count}/{config.RequiredResultsPerTerrain} 个符合条件的结果");
                    }
                    else
                    {
                        Debug.Log($"地形 {terrainId} 成功找到 {terrainValidResults.Count}/{config.RequiredResultsPerTerrain} 个符合条件的结果");
                    }
                }
            }

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
        /// 导出结果为CSV
        /// </summary>
        public static void ExportToCsv(List<AnalysisResult> results, string outputPath)
        {
            try
            {
                var csv = new StringBuilder();

                // CSV表头 - 添加难度分数字段
                csv.AppendLine("UniqueId,TerrainId,LevelName,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,RandomSeed," +
                              "GameCompleted,TotalMoves,GameDurationMs,CompletionStatus," +
                              "TotalAnalysisTimeMs,SuccessfulGroups,InitialMinCost," +
                              "DifficultyPosition,TileIdSequence,DockCountPerMove,PeakDockCount,DockAfterTrioMatch,SafeOptionCounts," +
                              "MinCostAfterTrioMatch,MinCostOptionsAfterTrioMatch,PressureValues," +
                              "PressureValueMean,PressureValueMin,PressureValueMax,PressureValueStdDev,DifficultyScore,FinalDifficulty,ErrorMessage");

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

                    csv.AppendLine($"{result.UniqueId},{result.TerrainId},{result.LevelName},{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.RandomSeed}," +
                                  $"{result.GameCompleted},{result.TotalMoves},{result.GameDurationMs},\"{result.CompletionStatus}\"," +
                                  $"{result.TotalAnalysisTimeMs},{result.SuccessfulGroups},{result.InitialMinCost}," +
                                  $"{result.DifficultyPosition:F4},\"{tileSequence}\",\"{dockCounts}\",{result.PeakDockCount},\"{dockAfterTrio}\",\"{safeOptions}\"," +
                                  $"\"{minCostAfterTrio}\",\"{minCostOptionsAfterTrio}\",\"{pressureValues}\"," +
                                  $"{result.PressureValueMean:F4},{result.PressureValueMin},{result.PressureValueMax},{result.PressureValueStdDev:F4},{result.DifficultyScore:F2},{result.FinalDifficulty},\"{result.ErrorMessage ?? ""}\"");
                }

                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                File.WriteAllText(outputPath, csv.ToString(), Encoding.UTF8);
                Debug.Log($"结果已导出到: {outputPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出CSV失败: {ex.Message}");
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
            Debug.Log($"关卡数量: {config.TestLevelCount}");

            var results = RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? $"_FixedSeeds" : "_Random";
            string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

            ExportToCsv(results, csvPath);

            if (config.IsFilteringEnabled)
            {
                Debug.Log($"筛选分析完成! 找到 {results.Count} 个符合条件的结果");
            }
            else
            {
                Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
            }
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹
            if (Directory.Exists(config.OutputDirectory))
            {
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
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

            GUILayout.Label("BattleAnalyzer 批量分析配置", EditorStyles.boldLabel);
            GUILayout.Space(10);

            // === CSV配置选择器 ===
            GUILayout.Label("CSV配置选择器", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.ExperienceConfigEnum = EditorGUILayout.IntField(new GUIContent("体验模式枚举",
                "1-6=exp-fix-1到exp-fix-6, -1=exp-range-1所有配置, -2=数组排列组合"), config.ExperienceConfigEnum);

            config.ColorCountConfigEnum = EditorGUILayout.IntField(new GUIContent("花色数量枚举",
                "1-6=type-count-1到type-count-6, -1=type-range-1所有配置, -2=动态范围"), config.ColorCountConfigEnum);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 测试参数 ===
            GUILayout.Label("测试参数", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.TestLevelCount = EditorGUILayout.IntField(new GUIContent("测试关卡数量", "要测试的关卡数量"), config.TestLevelCount);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 排列组合配置 ===
            GUILayout.Label("排列组合配置 (ExperienceConfigEnum = -2时生效)", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.ArrayLength = EditorGUILayout.IntField(new GUIContent("数组长度", "体验数组长度，如[a,b,c]为3"), config.ArrayLength);
            config.MinValue = EditorGUILayout.IntField(new GUIContent("最小值", "排列组合的最小值"), config.MinValue);
            config.MaxValue = EditorGUILayout.IntField(new GUIContent("最大值", "排列组合的最大值"), config.MaxValue);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 随机种子配置 ===
            GUILayout.Label("随机种子配置", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.UseFixedSeed = EditorGUILayout.Toggle(new GUIContent("使用固定种子", "true=结果可重现，false=完全随机"), config.UseFixedSeed);

            EditorGUILayout.LabelField("固定种子值列表（逗号分隔）:");
            seedValuesString = EditorGUILayout.TextArea(seedValuesString, GUILayout.Height(40));

            config.RunsPerLevel = EditorGUILayout.IntField(new GUIContent("每关卡运行次数", "每个关卡运行次数，用于生成多样化数据"), config.RunsPerLevel);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 筛选配置 ===
            GUILayout.Label("筛选配置", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.UseTerrainSpecificFiltering = EditorGUILayout.Toggle(new GUIContent("使用地形特定筛选", "从CSV读取position和score字段"), config.UseTerrainSpecificFiltering);
            config.EnableGlobalFiltering = EditorGUILayout.Toggle(new GUIContent("启用全局筛选", "作为fallback使用"), config.EnableGlobalFiltering);

            if (config.EnableGlobalFiltering)
            {
                EditorGUI.indentLevel++;
                config.GlobalDifficultyPositionRangeMin = EditorGUILayout.FloatField("全局难点位置范围最小值", config.GlobalDifficultyPositionRangeMin);
                config.GlobalDifficultyPositionRangeMax = EditorGUILayout.FloatField("全局难点位置范围最大值", config.GlobalDifficultyPositionRangeMax);
                config.GlobalDifficultyScoreRangeMin = EditorGUILayout.FloatField("全局难度分数范围最小值", config.GlobalDifficultyScoreRangeMin);
                config.GlobalDifficultyScoreRangeMax = EditorGUILayout.FloatField("全局难度分数范围最大值", config.GlobalDifficultyScoreRangeMax);
                EditorGUI.indentLevel--;
            }

            config.RequiredResultsPerTerrain = EditorGUILayout.IntField(new GUIContent("每地形需要结果数量", "每个地形需要找到的符合条件结果数量"), config.RequiredResultsPerTerrain);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 策略切换配置 ===
            GUILayout.Label("策略切换配置", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");

            config.EnableStrategicSwitching = EditorGUILayout.Toggle(new GUIContent("启用策略性配置切换", "是否启用策略性配置切换"), config.EnableStrategicSwitching);
            config.MaxConfigAttemptsPerTerrain = EditorGUILayout.IntField(new GUIContent("每地形最大尝试配置数量", "每个地形最大尝试配置数量"), config.MaxConfigAttemptsPerTerrain);

            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

            // === 输出配置 ===
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

            EditorGUILayout.EndVertical();
            GUILayout.Space(20);

            // === 预览和执行 ===
            GUILayout.Label("配置预览", EditorStyles.boldLabel);
            EditorGUILayout.BeginVertical("box");
            EditorGUILayout.LabelField("配置描述:", EditorStyles.wordWrappedLabel);
            EditorGUILayout.LabelField(config.GetConfigDescription(), EditorStyles.wordWrappedMiniLabel);
            EditorGUILayout.EndVertical();
            GUILayout.Space(10);

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
            // 解析种子值字符串
            ParseSeedValues();

            // 确保输出目录存在
            if (!Directory.Exists(config.OutputDirectory))
            {
                Directory.CreateDirectory(config.OutputDirectory);
            }

            Debug.Log($"=== 开始批量分析（自定义配置） ===");
            Debug.Log($"配置详情: {config.GetConfigDescription()}");
            Debug.Log($"关卡数量: {config.TestLevelCount}");

            var results = BattleAnalyzerRunner.RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? $"_FixedSeeds" : "_Random";
            string filterSuffix = config.IsFilteringEnabled ? "_Filtered" : "";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}{filterSuffix}_{timestamp}.csv");

            BattleAnalyzerRunner.ExportToCsv(results, csvPath);

            if (config.IsFilteringEnabled)
            {
                Debug.Log($"筛选分析完成! 找到 {results.Count} 个符合条件的结果");
            }
            else
            {
                Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
            }
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹
            if (Directory.Exists(config.OutputDirectory))
            {
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
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