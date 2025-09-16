using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using DGuo.Client.TileMatch;
using DGuo.Client;
using DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DGuo.Client.TileMatch.DesignerAlgo.Evaluation
{
    /// <summary>
    /// CSV配置数据行
    /// </summary>
    public class CsvConfigurationRow
    {
        public int TerrainId { get; set; }
        public int[] ExpFix1 { get; set; }
        public int[] ExpFix2 { get; set; }
        public int[] ExpRange1 { get; set; }
        public int TypeCount1 { get; set; }
        public int TypeCount2 { get; set; }
        public int TypeRange1 { get; set; }
    }

    /// <summary>
    /// 简化的批量评估配置管理器
    /// </summary>
    [System.Serializable]
    public class SimplifiedBatchConfiguration
    {
        [Header("=== CSV配置选择器 ===")]
        public int ExperienceConfigEnum = 1; // 体验模式枚举：1=exp-fix-1, 2=exp-fix-2, -1=exp-range-1所有, -2=exp-range-2所有
        public int ColorCountConfigEnum = 1; // 花色数量枚举：1=type-count-1, 2=type-count-2, -1=type-range-1所有, -2=type-range-2所有
        
        [Header("=== 测试参数 ===")]
        public int TestLevelCount = 15; // 测试关卡数量 - 修改这个数字选择测试多少个关卡
        
        [Header("=== 通用配置 ===")]
        public string[] PlayerTypesToEvaluate = { "Normal" };
        public string OutputDirectory = "DetailedResults";
        
    }

    /// <summary>
    /// CSV配置解析服务
    /// </summary>
    public static class CsvConfigurationResolver
    {
        private static Dictionary<int, CsvConfigurationRow> _csvData = null;
        
        /// <summary>
        /// 从CSV文件加载配置数据
        /// </summary>
        private static void LoadCsvData()
        {
            if (_csvData != null) return;
            
            _csvData = new Dictionary<int, CsvConfigurationRow>();
            
            try
            {
                string csvPath = Path.Combine(Application.dataPath, "_Editor/all_level.csv");
                if (!File.Exists(csvPath))
                {
                    Debug.LogError($"CSV配置文件不存在: {csvPath}");
                    return;
                }
                
                var lines = File.ReadAllLines(csvPath);
                for (int i = 1; i < lines.Length; i++) // 跳过表头
                {
                    var parts = ParseCsvLine(lines[i]);
                    if (parts.Length >= 7 && int.TryParse(parts[0], out int terrainId))
                    {
                        var row = new CsvConfigurationRow
                        {
                            TerrainId = terrainId,
                            ExpFix1 = ParseIntArray(parts[1]),
                            ExpFix2 = ParseIntArray(parts[2]),
                            ExpRange1 = ParseIntArray(parts[3]),
                            TypeCount1 = ParseIntOrDefault(parts[4], 7),
                            TypeCount2 = ParseIntOrDefault(parts[5], 8),
                            TypeRange1 = ParseIntOrDefault(parts[6], 7)
                        };
                        _csvData[terrainId] = row;
                    }
                }
                
                Debug.Log($"成功加载CSV配置数据，包含 {_csvData.Count} 个地形配置");
            }
            catch (Exception ex)
            {
                Debug.LogError($"加载CSV配置失败: {ex.Message}");
                _csvData = new Dictionary<int, CsvConfigurationRow>();
            }
        }
        
        /// <summary>
        /// 解析CSV行，处理引号包围的字段
        /// </summary>
        private static string[] ParseCsvLine(string line)
        {
            var result = new List<string>();
            var currentField = new StringBuilder();
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
                    result.Add(currentField.ToString());
                    currentField.Clear();
                }
                else
                {
                    currentField.Append(c);
                }
            }
            
            result.Add(currentField.ToString());
            return result.ToArray();
        }
        
        /// <summary>
        /// 解析整数数组字符串，支持多种格式
        /// 支持: "[1,2,3]", "1,2,3", "1 2 3", 单个数字等
        /// </summary>
        private static int[] ParseIntArray(string arrayStr)
        {
            try
            {
                if (string.IsNullOrEmpty(arrayStr))
                    return new int[] { 1, 2, 3 }; // 默认值
                
                // 去除空白字符和各种括号
                arrayStr = arrayStr.Trim().Trim('[', ']', '(', ')', '{', '}');
                
                if (string.IsNullOrEmpty(arrayStr))
                    return new int[] { 1, 2, 3 }; // 默认值
                
                // 尝试解析为单个数字
                if (int.TryParse(arrayStr, out int singleValue))
                {
                    return new int[] { singleValue, singleValue, singleValue };
                }
                
                // 分割字符串，支持逗号、空格、分号等分隔符
                var parts = arrayStr.Split(new char[] { ',', ' ', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                var result = new List<int>();
                
                foreach (var part in parts)
                {
                    if (int.TryParse(part.Trim(), out int value))
                    {
                        result.Add(value);
                    }
                }
                
                // 确保至少有3个值
                if (result.Count == 0)
                    return new int[] { 1, 2, 3 }; // 默认值
                else if (result.Count == 1)
                    return new int[] { result[0], result[0], result[0] };
                else if (result.Count == 2)
                    return new int[] { result[0], result[1], result[1] };
                else
                    return result.Take(3).ToArray(); // 只取前3个值
            }
            catch
            {
                return new int[] { 1, 2, 3 }; // 默认值
            }
        }
        
        /// <summary>
        /// 解析整数或返回默认值
        /// </summary>
        private static int ParseIntOrDefault(string str, int defaultValue)
        {
            return int.TryParse(str.Trim(), out int result) ? result : defaultValue;
        }
        
        /// <summary>
        /// 根据枚举配置解析体验模式
        /// </summary>
        public static int[][] ResolveExperienceModes(int experienceConfigEnum, string levelName)
        {
            LoadCsvData();
            
            switch (experienceConfigEnum)
            {
                case 1:
                case 2:
                    // 固定配置：使用特定地形的配置
                    int terrainId = ExtractLevelId(levelName);
                    if (!_csvData.ContainsKey(terrainId))
                        throw new System.Exception($"CSV中未找到地形ID {terrainId}");
                    
                    var row = _csvData[terrainId];
                    return new int[][] { experienceConfigEnum == 1 ? row.ExpFix1 : row.ExpFix2 };
                    
                case -1:
                    // 所有ExpRange1配置：返回全局去重后的所有配置
                    return GetAllExpRange1Configurations();
                    
                case -2:
                    // 所有ExpRange2配置：暂未定义，返回所有ExpRange1
                    return GetAllExpRange1Configurations();
                    
                default:
                    throw new System.Exception($"不支持的体验配置枚举: {experienceConfigEnum}");
            }
        }
        
        /// <summary>
        /// 根据枚举配置解析花色数量
        /// </summary>
        public static int[] ResolveColorCounts(int colorCountConfigEnum, string levelName)
        {
            LoadCsvData();
            
            switch (colorCountConfigEnum)
            {
                case 1:
                case 2:
                    // 固定配置：使用特定地形的配置
                    int terrainId = ExtractLevelId(levelName);
                    if (!_csvData.ContainsKey(terrainId))
                        throw new System.Exception($"CSV中未找到地形ID {terrainId}");
                    
                    var row = _csvData[terrainId];
                    return new int[] { colorCountConfigEnum == 1 ? row.TypeCount1 : row.TypeCount2 };
                    
                case -1:
                    // 所有TypeRange1配置：返回全局去重后的所有配置
                    return GetAllTypeRange1Configurations();
                    
                case -2:
                    // 所有TypeRange2配置：暂未定义，返回所有TypeRange1
                    return GetAllTypeRange1Configurations();
                    
                default:
                    throw new System.Exception($"不支持的花色配置枚举: {colorCountConfigEnum}");
            }
        }
        
        /// <summary>
        /// 获取exp-range-1列中所有不重复的体验配置
        /// </summary>
        private static int[][] GetAllExpRange1Configurations()
        {
            LoadCsvData();
            
            var uniqueConfigs = new HashSet<string>();
            var results = new List<int[]>();
            
            foreach (var row in _csvData.Values)
            {
                string configStr = string.Join(",", row.ExpRange1);
                if (uniqueConfigs.Add(configStr))
                {
                    results.Add(row.ExpRange1);
                }
            }
            
            return results.ToArray();
        }
        
        /// <summary>
        /// 获取type-range-1列中所有不重复的花色数量
        /// </summary>
        private static int[] GetAllTypeRange1Configurations()
        {
            LoadCsvData();
            
            return _csvData.Values
                .Select(row => row.TypeRange1)
                .Where(count => count > 0)
                .Distinct()
                .OrderBy(x => x)
                .ToArray();
        }

        /// <summary>
        /// 从Level_001格式提取关卡ID
        /// </summary>
        private static int ExtractLevelId(string levelName)
        {
            if (levelName.StartsWith("Level_"))
            {
                string numberPart = levelName.Substring(6); // 去掉"Level_"
                return int.Parse(numberPart);
            }
            return 1; // 默认关卡1
        }
    }

    /// <summary>
    /// 简化版批量关卡难度评估器 - 用于测试和演示
    /// 专注于基本功能，避免复杂的类型依赖问题
    /// </summary>
    public class BatchLevelEvaluatorSimple
    {
        /// <summary>
        /// 固定随机种子 - 确保评估结果可重现
        /// </summary>
        private const int FIXED_RANDOM_SEED = 12345678;

        // ========================================
        // 🎛️ 简化配置切换区域 - 修改这里选择你想要的配置
        // ========================================
        
        /// <summary>
        /// 动态复杂度分析开关 - true启用动态分析，false仅静态分析
        /// </summary>
        private static readonly bool ENABLE_DYNAMIC_ANALYSIS = true; // 默认启用动态分析（使用最优策略）
        
        /// <summary>
        /// 获取当前选择的配置
        /// </summary>
        public static SimplifiedBatchConfiguration GetSelectedConfig()
        {
            // 使用SimplifiedBatchConfiguration的默认实例配置
            return new SimplifiedBatchConfiguration();
        }
        
        /// <summary>
        /// 一键执行当前选择的配置 - 直接调用这个方法
        /// </summary>
        public static void RunSelectedConfiguration()
        {
            var config = GetSelectedConfig();
            string configName = $"CSV配置(体验{config.ExperienceConfigEnum}, 花色{config.ColorCountConfigEnum})";

            Debug.Log($"=== 正在执行{configName} ===");
            Debug.Log($"测试关卡数量: {config.TestLevelCount}个");
            Debug.Log($"体验模式枚举: {config.ExperienceConfigEnum}");
            Debug.Log($"花色数量枚举: {config.ColorCountConfigEnum}");


            ExecuteSimplifiedTest(config, configName, config.TestLevelCount);
        }
        
        // ========================================
        /// <summary>
        /// 详细的评估结果 - 包含8个指标值和加权计算，扩展支持动态复杂度数据
        /// </summary>
        [System.Serializable]
        public class DetailedEvaluationResult
        {
            public int UniqueID { get; set; }                                        // 唯一配置ID (顺序数字)
            public string LevelName { get; set; }                                    // 关卡名称 (JSON编号格式)
            public string Algorithm { get; set; }                                    // 算法类型
            public string ExperienceMode { get; set; }                               // 体验模式配置 (如[1,1,1])
            public string PlayerType { get; set; }                                   // 玩家类型
            public int ColorCount { get; set; }                                      // 花色数量 (7~14)
            public int TotalTiles { get; set; }                                      // 关卡总瓦片数量

            // 地形维度的3个指标值 (0-1)
            public float V_Normalized { get; set; }      // 归一化Tile总数
            public float E_Normalized { get; set; }      // 归一化暴露面
            public float A_Normalized { get; set; }      // 归一化堆叠层级

            // 花色维度的5个指标值 (0-1)
            public float C_Normalized { get; set; }      // 归一化花色数量
            public float D_Normalized { get; set; }      // 花色分布方差
            public float G_Normalized { get; set; }      // 平均同花色路径距离
            public float O_Normalized { get; set; }      // 花色暴露度差异
            public float M_Normalized { get; set; }      // 花色依赖深度差异

            // 加权计算结果
            public float TerrainScore { get; set; }      // 地形复杂度评分 (加权后)
            public float ColorScore { get; set; }        // 花色复杂度评分 (加权后)
            public float FinalScore { get; set; }        // 最终评分 (地形×系数 + 花色×系数)
            public int Grade { get; set; }               // 难度等级(数字: 1-5)

            public int ProcessingTimeMs { get; set; }    // 处理耗时(毫秒)
            public DateTime EvaluationTime { get; set; } // 评估时间
            public string ErrorMessage { get; set; }     // 错误信息(如果有)

            // 🆕 动态复杂度分析数据
            public bool DynamicAnalysisEnabled { get; set; } = true;                           // 是否启用动态分析
            public DynamicComplexityAnalyzer.DynamicComplexityMetrics DynamicMetrics { get; set; } = null;  // 动态指标数据
            public string DynamicAnalysisError { get; set; } = null;                            // 动态分析错误信息

            // 🆕 算法对比结果数据
            public DynamicComplexityAnalyzer.AlgorithmComparisonResult ComparisonResult { get; set; } = null;  // 算法对比结果
        }


        /// <summary>
        /// 批量评估进度信息
        /// </summary>
        public class BatchProgress
        {
            public int ProcessedTasks { get; set; }         // 已处理任务数
            public int TotalTasks { get; set; }             // 总任务数
            public string CurrentTask { get; set; }         // 当前任务描述
            public TimeSpan ElapsedTime { get; set; }       // 已耗时
            
            public float ProgressPercentage => TotalTasks > 0 ? (float)ProcessedTasks / TotalTasks * 100f : 0f;
            
            public override string ToString()
            {
                return $"进度: {ProcessedTasks}/{TotalTasks} ({ProgressPercentage:F1}%) | " +
                       $"当前: {CurrentTask} | " +
                       $"耗时: {ElapsedTime:hh\\:mm\\:ss}";
            }
        }

        /// <summary>
        /// 真实关卡评估 - 使用真实数据和算法计算评估结果
        /// </summary>
        /// <param name="levelName">关卡名称</param>
        /// <param name="experienceMode">体验模式配置 ([stage1,stage2,stage3])</param>
        /// <param name="playerType">玩家类型</param>
        /// <param name="colorCount">花色数量 (可选，默认使用关卡配置)</param>
        /// <param name="uniqueID">唯一配置ID</param>
        /// <returns>真实评估结果</returns>
        public static DetailedEvaluationResult EvaluateRealLevel(string levelName, int[] experienceMode, string playerType, int? colorCount = null, int uniqueID = 0)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            try
            {
                // 注意：随机种子已在批量评估开始时统一设置，此处无需重复设置
                
                // 第一步: 加载真实关卡数据
                var levelData = LoadLevelData(levelName);
                if (levelData == null)
                {
                    return CreateErrorResult(levelName, experienceMode, playerType, $"无法加载关卡数据: {levelName}", colorCount, uniqueID);
                }
                
                // 第二步: 创建Tile列表
                var tiles = CreateTileListFromLevelData(levelData);
                
                // 第三步: 运行RuleBased算法进行花色分配
                var algorithm = new RuleBasedAlgorithm();
                // 注意：随机种子已统一设置，算法将继承全局种子状态

                // 使用自定义花色数量或默认关卡配置
                int requestedColorCount = colorCount ?? levelData.ElementsPerLevel;
                var availableColors = CreateAvailableColors(requestedColorCount);
                int actualColorCount = availableColors.Count; // 实际使用的花色数量
                algorithm.AssignTileTypes(tiles, experienceMode, availableColors);

                // 获取实际算法名称
                string actualAlgorithmName = algorithm.AlgorithmName;

                // 第四步: 直接使用算法内部计算的详细评估结果，避免重复计算
                var detailedEvaluation = algorithm.LastDetailedEvaluationResult;

                if (detailedEvaluation == null)
                {
                    throw new System.Exception("算法内部的详细评估结果为空，可能算法执行失败");
                }

                // 第五步: 动态复杂度分析 - 对比两种算法 (可选)
                DynamicComplexityAnalyzer.DynamicComplexityMetrics dynamicMetrics = null;
                DynamicComplexityAnalyzer.AlgorithmComparisonResult comparisonResult = null;
                string dynamicError = null;
                bool dynamicEnabled = true;

                if (ENABLE_DYNAMIC_ANALYSIS)
                {
                    try
                    {
                        dynamicEnabled = true;

                        // 运行算法对比测试 - 同时运行两种算法
                        comparisonResult = DynamicComplexityAnalyzer.CompareAlgorithms(
                            tiles,
                            experienceMode,
                            algorithm.LastTerrainAnalysis,
                            DynamicComplexityAnalyzer.AlgorithmType.OptimalDFS,
                            DynamicComplexityAnalyzer.AlgorithmType.BattleAnalyzer
                        );

                        // 使用OptimalDFS的结果作为主要动态指标（向后兼容）
                        dynamicMetrics = comparisonResult.Algorithm1Metrics;

                        // 详细输出对比结果
                        if (!string.IsNullOrEmpty(comparisonResult.ErrorMessage))
                        {
                            Debug.LogWarning($"算法对比部分失败 {levelName}: {comparisonResult.ErrorMessage}");
                        }
                        else
                        {
                            var opt = comparisonResult.Algorithm1Metrics;
                            var battle = comparisonResult.Algorithm2Metrics;

                            Debug.Log($"=== 算法性能对比 {levelName} ===");
                            Debug.Log($"OptimalDFS算法: {opt?.CompletionStatus}, 移动{opt?.TotalMoves}步, 耗时{opt?.GameDurationMs}ms");
                            Debug.Log($"BattleAnalyzer算法: {battle?.CompletionStatus}, 移动{battle?.TotalMoves}步, 耗时{battle?.GameDurationMs}ms");
                            Debug.Log($"移动步数差异: {comparisonResult.MoveDifference} (优胜者: {comparisonResult.WinnerByMoves})");
                            Debug.Log($"执行时间差异: {comparisonResult.TimeDifference}ms (优胜者: {comparisonResult.WinnerByTime})");
                            Debug.Log($"结果一致性: {comparisonResult.SameResult}");

                            // 输出更详细的统计信息
                            if (opt != null)
                            {
                                var minPeak = opt.GetMetric<int>("MinPeakDock", -1);
                                var expandedNodes = opt.GetMetric<int>("ExpandedNodes", 0);
                                var visitedStates = opt.GetMetric<int>("VisitedStates", 0);
                                Debug.Log($"OptimalDFS详细: MinPeakDock={minPeak}, 扩展节点={expandedNodes}, 访问状态={visitedStates}");
                            }

                            if (battle != null)
                            {
                                var analysisCalls = battle.GetMetric<int>("TotalAnalysisCalls", 0);
                                var analysisTime = battle.GetMetric<int>("AnalysisTimeMs", 0);
                                var successMoves = battle.GetMetric<int>("SuccessfulMoves", 0);
                                Debug.Log($"BattleAnalyzer详细: 分析调用={analysisCalls}次, 分析耗时={analysisTime}ms, 成功移动={successMoves}次");
                            }
                        }
                    }
                    catch (Exception dynamicEx)
                    {
                        dynamicError = dynamicEx.Message;
                        Debug.LogWarning($"动态分析失败 {levelName}: {dynamicEx.Message}");
                    }
                }

                // 第六步: 直接从详细评估结果提取所有指标，无需重复计算
                var finalEval = detailedEvaluation.FinalEvaluation;
                var terrainComplexity = detailedEvaluation.TerrainComplexity;
                var colorComplexity = detailedEvaluation.ColorComplexity;
                int grade = (int)finalEval.Grade;

                // 转换关卡名称
                string jsonLevelName = ConvertToJsonLevelName(levelName);
                
                stopwatch.Stop();
                
                return new DetailedEvaluationResult
                {
                    UniqueID = uniqueID,
                    LevelName = jsonLevelName,
                    Algorithm = actualAlgorithmName,
                    ExperienceMode = $"[{experienceMode[0]},{experienceMode[1]},{experienceMode[2]}]",
                    PlayerType = playerType,
                    ColorCount = actualColorCount,
                    TotalTiles = CalculateTotalTileCount(levelData),

                    // 8个指标值 - 直接从算法内部计算的详细评估结果提取
                    V_Normalized = terrainComplexity.V_Normalized,
                    E_Normalized = terrainComplexity.E_Normalized,
                    A_Normalized = terrainComplexity.A_Normalized,
                    C_Normalized = colorComplexity.C_Normalized,
                    D_Normalized = colorComplexity.D_Normalized,
                    G_Normalized = colorComplexity.G_Normalized,
                    O_Normalized = colorComplexity.O_Normalized,
                    M_Normalized = colorComplexity.M_Normalized,

                    // 加权计算结果 - 直接从算法内部计算的最终评估结果提取
                    TerrainScore = finalEval.TerrainScore,
                    ColorScore = finalEval.ColorScore,
                    FinalScore = finalEval.FinalScore,
                    Grade = grade,

                    ProcessingTimeMs = (int)stopwatch.ElapsedMilliseconds,
                    EvaluationTime = DateTime.Now,
                    ErrorMessage = null,

                    // 动态复杂度分析结果
                    DynamicAnalysisEnabled = dynamicEnabled,
                    DynamicMetrics = dynamicMetrics,
                    DynamicAnalysisError = dynamicError,
                    ComparisonResult = comparisonResult
                };
            }
            catch (Exception ex)
            {
                stopwatch.Stop();
                string jsonLevelName = ConvertToJsonLevelName(levelName);
                
                return new DetailedEvaluationResult
                {
                    UniqueID = uniqueID,
                    LevelName = jsonLevelName,
                    Algorithm = "RuleBased-V1.1", // 使用默认算法名称（错误情况下）
                    ExperienceMode = $"[{experienceMode[0]},{experienceMode[1]},{experienceMode[2]}]",
                    PlayerType = playerType,
                    ColorCount = colorCount ?? 0,
                    TotalTiles = 0,
                    ErrorMessage = ex.Message,
                    ProcessingTimeMs = (int)stopwatch.ElapsedMilliseconds,
                    EvaluationTime = DateTime.Now,

                    // 🆕 错误情况下的动态分析状态
                    DynamicAnalysisEnabled = false,
                    DynamicMetrics = null,
                    DynamicAnalysisError = "Static analysis failed, dynamic analysis skipped",
                    ComparisonResult = null
                };
            }
        }

        /// <summary>
        /// 简化的批量评估主方法 - 使用新的CSV配置系统
        /// </summary>
        /// <param name="levelNames">关卡名称列表</param>
        /// <param name="config">简化评估配置</param>
        /// <param name="progressCallback">进度回调</param>
        /// <returns>详细评估结果列表</returns>
        public static List<DetailedEvaluationResult> EvaluateLevelsSimplified(
            List<string> levelNames,
            SimplifiedBatchConfiguration config = null,
            Action<BatchProgress> progressCallback = null)
        {
            if (config == null) config = new SimplifiedBatchConfiguration();

            // 设置固定随机种子确保批量评估结果可重现
            UnityEngine.Random.InitState(FIXED_RANDOM_SEED);
            Debug.Log($"批量评估使用固定随机种子: {FIXED_RANDOM_SEED}");
            Debug.Log($"体验模式枚举: {config.ExperienceConfigEnum}");
            Debug.Log($"花色数量枚举: {config.ColorCountConfigEnum}");

            var results = new List<DetailedEvaluationResult>();
            var startTime = DateTime.Now;
            int currentUniqueID = 1;
            int completedTasks = 0;

            // 🚀 优化：预计算所有关卡的配置，避免重复解析
            var levelConfigs = new Dictionary<string, (int[][] experienceModes, int[] colorCounts)>();
            int totalTasks = 0;

            foreach (var levelName in levelNames)
            {
                var experienceModes = CsvConfigurationResolver.ResolveExperienceModes(config.ExperienceConfigEnum, levelName);
                var colorCounts = CsvConfigurationResolver.ResolveColorCounts(config.ColorCountConfigEnum, levelName);
                levelConfigs[levelName] = (experienceModes, colorCounts);
                totalTasks += experienceModes.Length * config.PlayerTypesToEvaluate.Length * colorCounts.Length;
            }

            Debug.Log($"开始简化批量评估: {levelNames.Count} 个关卡 = {totalTasks} 个任务");

            // 执行评估 - 使用预计算的配置
            foreach (var levelName in levelNames)
            {
                var (experienceModes, colorCounts) = levelConfigs[levelName];

                foreach (var experienceMode in experienceModes)
                {
                    foreach (var playerType in config.PlayerTypesToEvaluate)
                    {
                        foreach (var colorCount in colorCounts)
                        {
                            // 执行评估
                            var result = EvaluateRealLevel(levelName, experienceMode, playerType, colorCount, currentUniqueID);
                            results.Add(result);
                            currentUniqueID++;
                            completedTasks++;

                            // 更新进度
                            var elapsed = DateTime.Now - startTime;
                            var taskName = $"{levelName}-[{experienceMode[0]},{experienceMode[1]},{experienceMode[2]}]-{playerType}-{colorCount}";
                            var progress = new BatchProgress
                            {
                                ProcessedTasks = completedTasks,
                                TotalTasks = totalTasks,
                                CurrentTask = taskName,
                                ElapsedTime = elapsed
                            };
                            progressCallback?.Invoke(progress);
                        }
                    }
                }
            }

            var totalTime = DateTime.Now - startTime;
            Debug.Log($"简化批量评估完成: 处理了 {results.Count} 个结果，总耗时 {totalTime:hh\\:mm\\:ss}");

            // 🚀 优化：批量评估完成后清理缓存，释放内存
            ClearAllCaches();

            return results;
        }


        /// <summary>
        /// 一键执行当前选择的配置 - 优先使用这个
        /// </summary>
        [UnityEditor.MenuItem("TileMatch/批量评估/▶️ 运行当前CSV配置")]
        public static void RunCurrentSelectedConfig()
        {
            RunSelectedConfiguration();
        }

        /// <summary>
        /// 简化的测试执行方法
        /// </summary>
        public static void ExecuteSimplifiedTest(SimplifiedBatchConfiguration config, string testName, int levelCount = 50)
        {
            try
            {
                Debug.Log($"=== 开始{testName}测试 ===");
                Debug.Log($"体验模式枚举: {config.ExperienceConfigEnum}");
                Debug.Log($"花色数量枚举: {config.ColorCountConfigEnum}");
                Debug.Log($"测试关卡数量: {levelCount}个");
                Debug.Log($"算法对比: 同时运行OptimalDFS和BattleAnalyzer两种算法");
                Debug.Log($"动态分析: {(ENABLE_DYNAMIC_ANALYSIS ? "启用" : "禁用")}");
                Debug.Log($"使用固定随机种子: {FIXED_RANDOM_SEED} (确保结果可重现)");
                
                // 创建测试关卡名称
                var levelNames = new List<string>();
                for (int i = 1; i <= levelCount; i++)
                {
                    levelNames.Add($"Level_{i:D3}");
                }
                
                config.OutputDirectory = Path.Combine(Application.dataPath, "_Editor/DetailedResults");

                var results = EvaluateLevelsSimplified(levelNames, config, progress => {
                    Debug.Log($"  {progress}");
                });

                Debug.Log($"{testName}测试完成！生成了 {results.Count} 个评估结果");

                // 导出结果
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var csvPath = Path.Combine(config.OutputDirectory, $"CsvConfig_{timestamp}.csv");

                ExportDetailedToCsv(results, csvPath);

                Debug.Log($"结果已导出到: {config.OutputDirectory}");
                Debug.Log($"CSV文件: {csvPath}");

                // 打开输出文件夹
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
            }
            catch (Exception ex)
            {
                Debug.LogError($"{testName}测试失败: {ex.Message}\n{ex.StackTrace}");
            }
        }

        /// <summary>
        /// CSV导出器 - 统一处理CSV格式化需求
        /// </summary>
        private static class CsvExporter
        {
            public const string CSV_HEADER = "UniqueID,LevelName,Algorithm,ExperienceMode,PlayerType,ColorCount,TotalTiles," +
                                             "V_Normalized,E_Normalized,A_Normalized," +
                                             "C_Normalized,D_Normalized,G_Normalized,O_Normalized,M_Normalized," +
                                             "TerrainScore,ColorScore,FinalScore,Grade," +
                                             "ProcessingTimeMs,EvaluationTime,ErrorMessage," +
                                             "DynamicAnalysisEnabled,TotalMoves,GameDurationMs,GameCompleted,CompletionStatus,DynamicAnalysisError," +
                                             "MinPeakDock,ExpandedNodes,VisitedStates,SolveTimeMs," +
                                             "OptimalTileIdSequence,DockCountPerMove,PeakDockDuringSolution," +
                                             "BattleAnalyzer_TotalMoves,BattleAnalyzer_GameDurationMs,BattleAnalyzer_CompletionStatus," +
                                             "BattleAnalyzer_AnalysisCalls,BattleAnalyzer_AnalysisTimeMs,BattleAnalyzer_SuccessfulMoves," +
                                             "MoveDifference,TimeDifference,WinnerByMoves,WinnerByTime,SameResult";

            /// <summary>
            /// CSV字段转义 - 内联工具方法
            /// </summary>
            private static string Escape(string field) =>
                string.IsNullOrEmpty(field) ? "" :
                (field.Contains(",") || field.Contains("\"") || field.Contains("\n") || field.Contains("\r")) ?
                "\"" + field.Replace("\"", "\"\"") + "\"" : field;

            /// <summary>
            /// 格式化动态分析字段（包含对比数据）
            /// </summary>
            public static string FormatDynamicFields(DetailedEvaluationResult result)
            {
                if (result.DynamicAnalysisEnabled && result.DynamicMetrics != null)
                {
                    int minPeak = result.DynamicMetrics.GetMetric<int>("MinPeakDock", -1);
                    int expanded = result.DynamicMetrics.GetMetric<int>("ExpandedNodes", 0);
                    int visited = result.DynamicMetrics.GetMetric<int>("VisitedStates", 0);
                    int solveMs = result.DynamicMetrics.GetMetric<int>("SolveTimeMs", 0);

                    // 序列化最优解序列与每步Dock数量（用空格分隔，并整体加引号）
                    string seq = (result.DynamicMetrics.OptimalMoveTileIds != null && result.DynamicMetrics.OptimalMoveTileIds.Count > 0)
                        ? "[" + string.Join(" ", result.DynamicMetrics.OptimalMoveTileIds) + "]"
                        : "";
                    string docks = (result.DynamicMetrics.DockCountPerMove != null && result.DynamicMetrics.DockCountPerMove.Count > 0)
                        ? "[" + string.Join(" ", result.DynamicMetrics.DockCountPerMove) + "]"
                        : "";
                    int peakDock = result.DynamicMetrics.PeakDockDuringSolution;

                    // BattleAnalyzer对比数据
                    string battleAnalyzerFields = "";
                    if (result.ComparisonResult != null && result.ComparisonResult.Algorithm2Metrics != null)
                    {
                        var battle = result.ComparisonResult.Algorithm2Metrics;
                        var analysisCalls = battle.GetMetric<int>("TotalAnalysisCalls", 0);
                        var analysisTime = battle.GetMetric<int>("AnalysisTimeMs", 0);
                        var successMoves = battle.GetMetric<int>("SuccessfulMoves", 0);

                        battleAnalyzerFields = string.Join(",",
                            battle.TotalMoves.ToString(),
                            battle.GameDurationMs.ToString(),
                            Escape(battle.CompletionStatus ?? ""),
                            analysisCalls.ToString(),
                            analysisTime.ToString(),
                            successMoves.ToString(),
                            result.ComparisonResult.MoveDifference.ToString(),
                            result.ComparisonResult.TimeDifference.ToString(),
                            Escape(result.ComparisonResult.WinnerByMoves ?? ""),
                            Escape(result.ComparisonResult.WinnerByTime ?? ""),
                            result.ComparisonResult.SameResult.ToString()
                        );
                    }
                    else
                    {
                        // 无对比数据时填充空值
                        battleAnalyzerFields = "0,0,,0,0,0,0,0,,,False";
                    }

                    return string.Join(",",
                        "True",
                        result.DynamicMetrics.TotalMoves.ToString(),
                        result.DynamicMetrics.GameDurationMs.ToString(),
                        result.DynamicMetrics.GameCompleted.ToString(),
                        Escape(result.DynamicMetrics.CompletionStatus ?? ""),
                        Escape(result.DynamicAnalysisError ?? ""),
                        minPeak.ToString(), expanded.ToString(), visited.ToString(), solveMs.ToString(),
                        Escape(seq), Escape(docks), peakDock.ToString(),
                        battleAnalyzerFields
                    );
                }
                return string.Join(",", "False", "0", "0", "False",
                    Escape("Not_Analyzed"),
                    Escape(result.DynamicAnalysisError ?? "Dynamic_Analysis_Disabled"),
                    "-1", "0", "0", "0",
                    "", "", "0",
                    "0", "0", "", "0", "0", "0", "0", "0", "", "", "False");
            }

            /// <summary>
            /// 格式化单行数据
            /// </summary>
            public static string FormatRow(DetailedEvaluationResult result)
            {
                return string.Join(",",
                    // 基础信息
                    result.UniqueID.ToString(),
                    Escape(result.LevelName),
                    Escape(result.Algorithm),
                    Escape(result.ExperienceMode),
                    Escape(result.PlayerType),
                    result.ColorCount.ToString(),
                    result.TotalTiles.ToString(),
                    // 8个指标值
                    result.V_Normalized.ToString("F6"), result.E_Normalized.ToString("F6"), result.A_Normalized.ToString("F6"),
                    result.C_Normalized.ToString("F6"), result.D_Normalized.ToString("F6"), result.G_Normalized.ToString("F6"),
                    result.O_Normalized.ToString("F6"), result.M_Normalized.ToString("F6"),
                    // 加权结果
                    result.TerrainScore.ToString("F2"), result.ColorScore.ToString("F2"), result.FinalScore.ToString("F2"),
                    result.Grade.ToString(),
                    // 元数据
                    result.ProcessingTimeMs.ToString(),
                    result.EvaluationTime.ToString("yyyy-MM-dd HH:mm:ss"),
                    Escape(result.ErrorMessage ?? ""),
                    // 动态分析
                    FormatDynamicFields(result)
                );
            }
        }

        /// <summary>
        /// 导出详细评估结果为CSV格式 - 优化版
        /// </summary>
        public static void ExportDetailedToCsv(List<DetailedEvaluationResult> results, string outputPath)
        {
            try
            {
                var csv = new StringBuilder();
                csv.AppendLine(CsvExporter.CSV_HEADER);

                foreach (var result in results)
                {
                    csv.AppendLine(CsvExporter.FormatRow(result));
                }

                // 确保输出目录存在并写入文件
                EnsureDirectoryExists(outputPath);
                File.WriteAllText(outputPath, csv.ToString(), Encoding.UTF8);

                Debug.Log($"详细CSV文件导出成功: {outputPath}, 包含 {results.Count} 条记录");
            }
            catch (Exception ex)
            {
                Debug.LogError($"导出详细CSV文件失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 确保目录存在
        /// </summary>
        private static void EnsureDirectoryExists(string filePath)
        {
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
        }

        /// <summary>
        /// 创建错误结果
        /// </summary>
        private static DetailedEvaluationResult CreateErrorResult(string levelName, int[] experienceMode, string playerType, string errorMessage, int? colorCount = null, int uniqueID = 0)
        {
            string jsonLevelName = ConvertToJsonLevelName(levelName);
            
            return new DetailedEvaluationResult
            {
                UniqueID = uniqueID,
                LevelName = jsonLevelName,
                Algorithm = "RuleBased-V1.1", // 使用默认算法名称
                ExperienceMode = $"[{experienceMode[0]},{experienceMode[1]},{experienceMode[2]}]",
                PlayerType = playerType,
                ColorCount = colorCount ?? 0,
                TotalTiles = 0,
                ProcessingTimeMs = 0,
                EvaluationTime = DateTime.Now,
                ErrorMessage = errorMessage
            };
        }

        /// <summary>
        /// 缓存的关卡数据 - 避免重复文件I/O
        /// </summary>
        private static Dictionary<string, LevelData> _cachedLevelData = new Dictionary<string, LevelData>();

        /// <summary>
        /// 清理所有缓存 - 释放内存，通常在批量评估完成后调用
        /// </summary>
        public static void ClearAllCaches()
        {
            _cachedLevelData?.Clear();
            _cachedAvailableElementValues = null;
            Debug.Log("[BatchLevelEvaluator] 所有缓存已清理");
        }

        /// <summary>
        /// 加载关卡数据 - 带缓存的从JSON文件加载
        /// </summary>
        private static LevelData LoadLevelData(string levelName)
        {
            // 🚀 优化：使用缓存避免重复文件I/O
            if (_cachedLevelData.TryGetValue(levelName, out var cachedData))
            {
                return cachedData;
            }

            try
            {
                // 根据levelName生成对应的JSON文件名
                string jsonFileName;
                if (levelName.StartsWith("Level_"))
                {
                    // Level_001 -> 100001.json
                    string numberPart = levelName.Substring(6); // 去掉"Level_"
                    int levelNumber = int.Parse(numberPart);
                    jsonFileName = $"{100000 + levelNumber}.json";
                }
                else
                {
                    // 直接使用levelName作为文件名
                    jsonFileName = levelName.EndsWith(".json") ? levelName : $"{levelName}.json";
                }

                // 构造JSON文件路径
                string jsonPath = Path.Combine(Application.dataPath, "..", "Tools", "Config", "Json", "Levels", jsonFileName);
                jsonPath = Path.GetFullPath(jsonPath); // 规范化路径

                if (!File.Exists(jsonPath))
                {
                    Debug.LogError($"关卡JSON文件不存在: {jsonPath}");
                    _cachedLevelData[levelName] = null; // 缓存失败结果，避免重复尝试
                    return null;
                }

                // 读取JSON文件内容
                string jsonContent = File.ReadAllText(jsonPath);

                // 反序列化为LevelData对象
                LevelData levelData = JsonUtility.FromJson<LevelData>(jsonContent);

                // 缓存成功加载的数据
                _cachedLevelData[levelName] = levelData;
                Debug.Log($"成功加载并缓存关卡数据: {levelName} -> {jsonFileName}");
                return levelData;
            }
            catch (Exception ex)
            {
                Debug.LogError($"加载关卡数据失败 {levelName}: {ex.Message}");
                _cachedLevelData[levelName] = null; // 缓存失败结果，避免重复尝试
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
                    
                    // 如果是固定类型，设置ElementValue
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
        /// 创建可用花色列表
        /// </summary>
        private static List<int> CreateAvailableColors(int colorCount)
        {
            // 动态读取可用花色池
            int[] availableColors = GetAvailableElementValues();
            
            if (colorCount <= availableColors.Length)
            {
                // 随机选择指定数量的花色
                var shuffled = new List<int>(availableColors);
                for (int i = 0; i < shuffled.Count; i++)
                {
                    int randomIndex = UnityEngine.Random.Range(i, shuffled.Count);
                    (shuffled[i], shuffled[randomIndex]) = (shuffled[randomIndex], shuffled[i]);
                }
                return shuffled.GetRange(0, colorCount);
            }
            else
            {
                // 使用所有可用花色
                return new List<int>(availableColors);
            }
        }

        /// <summary>
        /// 缓存的ElementValue数组 - 避免重复加载Resources
        /// </summary>
        private static int[] _cachedAvailableElementValues = null;

        /// <summary>
        /// 获取可用的ElementValue数组 - 带缓存的动态读取LevelDatabase
        /// </summary>
        private static int[] GetAvailableElementValues()
        {
            // 🚀 优化：使用缓存避免重复加载Resources
            if (_cachedAvailableElementValues != null)
            {
                return _cachedAvailableElementValues;
            }

            try
            {
                // 尝试加载LevelDatabase
                var levelDatabase = UnityEngine.Resources.Load<LevelDatabase>("StaticSettings/LevelDatabase");
                if (levelDatabase != null && levelDatabase.Tiles != null)
                {
                    _cachedAvailableElementValues = levelDatabase.Tiles
                        .Where(tile => tile != null && tile.ElementValue > 0)
                        .Select(tile => tile.ElementValue)
                        .Distinct()
                        .OrderBy(x => x)
                        .ToArray();

                    Debug.Log($"[BatchLevelEvaluator] 成功加载LevelDatabase，发现 {_cachedAvailableElementValues.Length} 种可用花色并已缓存");
                    return _cachedAvailableElementValues;
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[BatchLevelEvaluator] 无法加载LevelDatabase: {ex.Message}");
            }

            // 回退到标准花色池并缓存
            _cachedAvailableElementValues = new int[] { 101, 102, 103, 201, 202, 301, 302, 401, 402, 403, 501, 502, 601, 602, 701, 702, 703, 801, 802 };
            Debug.Log($"[BatchLevelEvaluator] 使用标准花色池，共 {_cachedAvailableElementValues.Length} 种花色并已缓存");
            return _cachedAvailableElementValues;
        }

        /// <summary>
        /// 将Level_001格式转换为JSON文件编号格式 (如100001)
        /// </summary>
        private static string ConvertToJsonLevelName(string levelName)
        {
            if (levelName.StartsWith("Level_"))
            {
                // Level_001 -> 100001
                string numberPart = levelName.Substring(6); // 去掉"Level_"
                if (int.TryParse(numberPart, out int levelNumber))
                {
                    return (100000 + levelNumber).ToString();
                }
            }
            return levelName; // 如果格式不匹配，返回原名称
        }

        /// <summary>
        /// 计算关卡总瓦片数量
        /// </summary>
        private static int CalculateTotalTileCount(LevelData levelData)
        {
            int totalCount = 0;
            foreach (var layer in levelData.Layers)
            {
                totalCount += layer.tiles.Length;
            }
            return totalCount;
        }
        
        
    }
}
