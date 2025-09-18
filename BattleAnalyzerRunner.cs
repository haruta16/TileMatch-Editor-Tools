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
            public int[] ExpRange1 { get; set; }
            public int TypeCount1 { get; set; }
            public int TypeCount2 { get; set; }
            public int TypeRange1 { get; set; }
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

            public string ErrorMessage { get; set; }
        }

        /// <summary>
        /// 批量运行配置 - 参考BatchLevelEvaluatorSimple的配置模式
        /// </summary>
        [System.Serializable]
        public class RunConfig
        {
            [Header("=== CSV配置选择器 ===")]
            public int ExperienceConfigEnum = 1; // 体验模式枚举：1=exp-fix-1, 2=exp-fix-2, -1=exp-range-1所有配置
            public int ColorCountConfigEnum = 1; // 花色数量枚举：1=type-count-1, 2=type-count-2, -1=type-range-1所有配置

            [Header("=== 测试参数 ===")]
            public int TestLevelCount = 50; // 测试关卡数量

            [Header("=== 随机种子配置 ===")]
            public bool UseFixedSeed = true; // 是否使用固定种子：true=结果可重现，false=完全随机
            public int FixedSeedValue = 12345678; // 固定种子值（当UseFixedSeed=true时使用）
            public int RunsPerLevel = 1; // 每个关卡运行次数：用于生成多样化数据

            [Header("=== 输出配置 ===")]
            public string OutputDirectory = "BattleAnalysisResults";

            /// <summary>
            /// 获取用于测试的随机种子
            /// </summary>
            public int GetSeedForRun(int levelIndex, int runIndex)
            {
                if (UseFixedSeed)
                {
                    // 固定种子模式：基于关卡和运行索引生成确定性种子
                    return FixedSeedValue + levelIndex * 1000 + runIndex;
                }
                else
                {
                    // 随机种子模式：生成完全随机的种子
                    return UnityEngine.Random.Range(1, int.MaxValue);
                }
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
                    -1 => "所有ExpRange1配置",
                    -2 => "所有ExpRange2配置",
                    _ => $"配置{ExperienceConfigEnum}"
                };

                string colorMode = ColorCountConfigEnum switch
                {
                    1 => "TypeCount1",
                    2 => "TypeCount2",
                    -1 => "所有TypeRange1配置",
                    -2 => "所有TypeRange2配置",
                    _ => $"配置{ColorCountConfigEnum}"
                };

                string seedMode = UseFixedSeed ? $"固定种子({FixedSeedValue})" : "随机种子";
                return $"体验模式[{expMode}], 花色数量[{colorMode}], {seedMode}, 每关卡{RunsPerLevel}次";
            }
        }

        private static Dictionary<int, CsvLevelConfig> _csvConfigs = null;

        /// <summary>
        /// CSV配置管理器 - 统一的配置加载和解析服务
        /// </summary>
        public static class CsvConfigManager
        {
            /// <summary>
            /// 加载CSV配置数据
            /// </summary>
            public static void LoadCsvConfigs()
            {
                if (_csvConfigs != null) return;

                _csvConfigs = new Dictionary<int, CsvLevelConfig>();

                try
                {
                    string csvPath = Path.Combine(Application.dataPath, "_Editor/all_level.csv");
                    if (!File.Exists(csvPath))
                    {
                        Debug.LogError($"CSV配置文件不存在: {csvPath}");
                        return;
                    }

                    var lines = File.ReadAllLines(csvPath, Encoding.UTF8);
                    for (int i = 1; i < lines.Length; i++) // 跳过表头
                    {
                        var parts = CsvParser.ParseCsvLine(lines[i]);
                        if (parts.Length >= 7 && int.TryParse(parts[0], out int terrainId))
                        {
                            var config = new CsvLevelConfig
                            {
                                TerrainId = terrainId,
                                ExpFix1 = CsvParser.ParseIntArray(parts[1]),
                                ExpFix2 = CsvParser.ParseIntArray(parts[2]),
                                ExpRange1 = CsvParser.ParseIntArray(parts[3]),
                                TypeCount1 = CsvParser.ParseIntOrDefault(parts[4], 7),
                                TypeCount2 = CsvParser.ParseIntOrDefault(parts[5], 8),
                                TypeRange1 = CsvParser.ParseIntOrDefault(parts[6], 7)
                            };
                            _csvConfigs[terrainId] = config;
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

            /// <summary>
            /// 根据枚举配置解析体验模式数组 - 支持-1全配置模式
            /// </summary>
            public static int[][] ResolveExperienceModes(int experienceConfigEnum, int terrainId)
            {
                LoadCsvConfigs();

                switch (experienceConfigEnum)
                {
                    case 1:
                    case 2:
                        // 固定配置：使用特定地形的配置
                        if (!_csvConfigs.TryGetValue(terrainId, out var config))
                        {
                            Debug.LogWarning($"未找到地形ID {terrainId} 的配置，使用默认值");
                            return new int[][] { new int[] { 1, 2, 3 } };
                        }

                        var selectedMode = experienceConfigEnum == 1 ? config.ExpFix1 : config.ExpFix2;
                        return new int[][] { selectedMode };

                    case -1:
                        // 所有ExpRange1配置：返回全局去重后的所有配置
                        return GetAllExpRange1Configurations();

                    case -2:
                        // 所有ExpRange2配置：暂未定义，返回所有ExpRange1
                        return GetAllExpRange1Configurations();

                    default:
                        Debug.LogWarning($"不支持的体验配置枚举: {experienceConfigEnum}，使用默认值");
                        return new int[][] { new int[] { 1, 2, 3 } };
                }
            }

            /// <summary>
            /// 根据枚举配置解析花色数量数组 - 支持-1全配置模式
            /// </summary>
            public static int[] ResolveColorCounts(int colorCountConfigEnum, int terrainId)
            {
                LoadCsvConfigs();

                switch (colorCountConfigEnum)
                {
                    case 1:
                    case 2:
                        // 固定配置：使用特定地形的配置
                        if (!_csvConfigs.TryGetValue(terrainId, out var config))
                        {
                            Debug.LogWarning($"未找到地形ID {terrainId} 的配置，使用默认值");
                            return new int[] { 7 };
                        }

                        var selectedCount = colorCountConfigEnum == 1 ? config.TypeCount1 : config.TypeCount2;
                        return new int[] { selectedCount };

                    case -1:
                        // 所有TypeRange1配置：返回全局去重后的所有配置
                        return GetAllTypeRange1Configurations();

                    case -2:
                        // 所有TypeRange2配置：暂未定义，返回所有TypeRange1
                        return GetAllTypeRange1Configurations();

                    default:
                        Debug.LogWarning($"不支持的花色配置枚举: {colorCountConfigEnum}，使用默认值");
                        return new int[] { 7 };
                }
            }

            /// <summary>
            /// 获取exp-range-1列中所有不重复的体验配置
            /// </summary>
            private static int[][] GetAllExpRange1Configurations()
            {
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

                Debug.Log($"获取到 {results.Count} 个不重复的ExpRange1配置");
                return results.ToArray();
            }

            /// <summary>
            /// 获取type-range-1列中所有不重复的花色数量
            /// </summary>
            private static int[] GetAllTypeRange1Configurations()
            {
                LoadCsvConfigs();

                var results = _csvConfigs.Values
                    .Select(row => row.TypeRange1)
                    .Where(count => count > 0)
                    .Distinct()
                    .OrderBy(x => x)
                    .ToArray();

                Debug.Log($"获取到 {results.Length} 个不重复的TypeRange1配置: [{string.Join(",", results)}]");
                return results;
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
        }

        /// <summary>
        /// CSV解析工具类 - 提取通用解析逻辑
        /// </summary>
        public static class CsvParser
        {
            /// <summary>
            /// 解析CSV行，处理引号包围的字段
            /// </summary>
            public static string[] ParseCsvLine(string line)
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
                        return result.Take(3).ToArray();
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
                return int.TryParse(str.Trim(), out int result) ? result : defaultValue;
            }
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

                // 使用RuleBasedAlgorithm进行花色分配
                var algorithm = new DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo.RuleBasedAlgorithm();
                algorithm.AssignTileTypes(tiles, experienceMode, availableColors);

                // 获取真实使用的算法名称
                result.AlgorithmName = algorithm.AlgorithmName;

                // 3. 创建虚拟对局环境
                var tileDict = tiles.ToDictionary(t => t.ID);
                var elementValues = tiles.Select(t => t.ElementValue).Where(v => v > 0).Distinct().ToList();

                // 4. 模拟自动游戏过程
                result = SimulateAutoPlay(tileDict, elementValues, result);

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

        /// <summary>
        /// 模拟自动游戏过程 - 完全复现真实游戏逻辑
        /// </summary>
        private static AnalysisResult SimulateAutoPlay(Dictionary<int, Tile> allTiles, List<int> elementValues, AnalysisResult result)
        {
            // 初始化状态
            var dockTiles = new List<Tile>();
            int moveCount = 0;
            int analysisTimeMs = 0;
            int SuccessfulGroups = 0;
            int peakDockCount = 0;

            // 初始化所有瓦片状态
            foreach (var tile in allTiles.Values)
            {
                tile.PileType = PileType.Desk;
                tile.ClearFlag();
            }

            // 创建虚拟分析器并设置花色
            var virtualAnalyzer = new VirtualBattleAnalyzer(allTiles, elementValues);

            // 游戏主循环
            while (true)
            {
                // 1. 更新瓦片状态 - 完全复现UpdateTilesState逻辑
                UpdateTilesStateCorrectly(allTiles);

                // 2. 分析当前局面 - 完全复现AnalyzerMgr.Analyze逻辑
                var analysisStopwatch = System.Diagnostics.Stopwatch.StartNew();
                virtualAnalyzer.Analyze();
                analysisStopwatch.Stop();
                analysisTimeMs += (int)analysisStopwatch.ElapsedMilliseconds;

                // 在第一次循环时计算游戏开局的最小cost值
                if (moveCount == 0)
                {
                    var (initialMinCost, _) = CalculateMinCostAndOptionsFromExisting(virtualAnalyzer);
                    result.InitialMinCost = initialMinCost;
                }

                // 3. 获取可点击的瓦片
                var clickableTiles = allTiles.Values
                    .Where(t => t.PileType == PileType.Desk && t.IsClickable)
                    .ToList();

                // 记录安全选项数量（每步都记录，dock<4时记录0）
                int safeOptionCount = dockTiles.Count >= 4 ? CountSafeOptions(virtualAnalyzer, dockTiles.Count) : 0;
                result.SafeOptionCounts.Add(safeOptionCount);

                // 4. 检查胜利条件
                if (clickableTiles.Count == 0 && allTiles.Values.All(t => t.PileType != PileType.Desk))
                {
                    result.GameCompleted = true;
                    result.CompletionStatus = "Success";
                    break;
                }

                // 5. 检查失败条件
                if (clickableTiles.Count == 0 && dockTiles.Count >= 7)
                {
                    result.CompletionStatus = "Dock_Full_Game_Over";
                    break;
                }

                if (clickableTiles.Count == 0)
                {
                    result.CompletionStatus = "No_Valid_Moves";
                    break;
                }

                // 6. 使用完全相同的AutoPlay逻辑选择最佳瓦片
                var selectedTile = SelectBestTileUsingRealAutoPlayLogic(clickableTiles, virtualAnalyzer);
                if (selectedTile == null)
                {
                    result.CompletionStatus = "AutoPlay_Strategy_Failed";
                    break;
                }

                // 7. 执行移动
                moveCount++;
                selectedTile.PileType = PileType.Dock;
                selectedTile.RemoveFlag(ETileFlag.Clickable);
                dockTiles.Add(selectedTile);
                result.TileIdSequence.Add(selectedTile.ID);
                result.DockCountPerMove.Add(dockTiles.Count);

                // 更新峰值Dock数量
                peakDockCount = Math.Max(peakDockCount, dockTiles.Count);

                // 8. 检查并执行消除
                bool matchOccurred = false;

                while (true)
                {
                    var matchedTiles = FindMatchInDock(dockTiles);
                    if (matchedTiles == null || matchedTiles.Count != 3)
                        break;

                    // 消除三个相同花色的瓦片
                    foreach (var tile in matchedTiles)
                    {
                        dockTiles.Remove(tile);
                        tile.SetFlag(ETileFlag.Destroyed);
                        tile.PileType = PileType.Discard;
                    }
                    SuccessfulGroups++;
                    matchOccurred = true;

                    // 记录三消后的dock数量
                    result.DockAfterTrioMatch.Add(dockTiles.Count);

                    // 三消完成后，重新分析并记录cost信息
                    virtualAnalyzer.Analyze();
                    var (postTrioCost, postTrioOptions) = CalculateMinCostAndOptionsFromExisting(virtualAnalyzer);
                    result.MinCostAfterTrioMatch.Add(postTrioCost);
                    result.MinCostOptionsAfterTrioMatch.Add(postTrioOptions);
                }

                // 9. 检查死亡条件
                if (dockTiles.Count >= 7)
                {
                    result.CompletionStatus = "Dock_Full_Game_Over";
                    break;
                }

                // 10. 检查无限循环保护
                if (moveCount > 10000)
                {
                    result.CompletionStatus = "Max_Moves_Exceeded";
                    break;
                }
            }

            // 设置结果
            result.TotalMoves = moveCount;
            result.TotalAnalysisTimeMs = analysisTimeMs;
            result.SuccessfulGroups = SuccessfulGroups;
            result.PeakDockCount = peakDockCount;
            result.MinMovesToComplete = result.TileIdSequence.Count;

            // 计算难点位置
            result.DifficultyPosition = CalculateDifficultyPosition(result.DockCountPerMove, peakDockCount);

            if (string.IsNullOrEmpty(result.CompletionStatus))
            {
                result.CompletionStatus = "Incomplete";
            }

            return result;
        }

        /// <summary>
        /// 计算安全选项数量：拥有全局最小cost且cost + dock <= 7的组合总数量
        /// </summary>
        private static int CountSafeOptions(VirtualBattleAnalyzer analyzer, int currentDockCount)
        {
            var allMatchGroups = new List<TileMatchBattleAnalyzerMgr.MatchGroup>();

            // 收集所有花色的所有组合
            foreach (var elementValue in analyzer.ElementValues)
            {
                var matchGroups = analyzer.GetMatchGroups(elementValue);
                allMatchGroups.AddRange(matchGroups);
            }

            if (allMatchGroups.Count == 0)
                return 0;

            // 找到全局最小cost
            int globalMinCost = allMatchGroups.Min(g => g.totalCost);

            // 统计最小cost且安全的组合数量
            int safeCount = 0;
            foreach (var matchGroup in allMatchGroups)
            {
                if (matchGroup.totalCost == globalMinCost && matchGroup.totalCost + currentDockCount <= 7)
                {
                    safeCount++;
                }
            }

            return safeCount;
        }

        /// <summary>
        /// 计算难点位置：基于peakdock在关卡进度中的位置（0~1）
        /// </summary>
        private static double CalculateDifficultyPosition(List<int> dockCountPerMove, int peakDockCount)
        {
            if (dockCountPerMove.Count == 0)
                return 0.0;

            // 找到所有等于peakdock的位置（从0开始的索引）
            var peakPositions = new List<int>();
            for (int i = 0; i < dockCountPerMove.Count; i++)
            {
                if (dockCountPerMove[i] == peakDockCount)
                {
                    peakPositions.Add(i);
                }
            }

            if (peakPositions.Count == 0)
                return 0.0;

            // 如果只有一个峰值位置，直接返回
            if (peakPositions.Count == 1)
            {
                return (double)peakPositions[0] / (dockCountPerMove.Count - 1);
            }

            // 多个峰值位置时，比较周围区域的dock均值，选择更高的
            int bestPosition = peakPositions[0];
            double bestSurroundingAverage = 0.0;

            foreach (int position in peakPositions)
            {
                double surroundingAverage = CalculateSurroundingAverage(dockCountPerMove, position);
                if (surroundingAverage > bestSurroundingAverage)
                {
                    bestSurroundingAverage = surroundingAverage;
                    bestPosition = position;
                }
            }

            return (double)bestPosition / (dockCountPerMove.Count - 1);
        }

        /// <summary>
        /// 计算指定位置周围区域的dock均值（窗口大小为5）
        /// </summary>
        private static double CalculateSurroundingAverage(List<int> dockCountPerMove, int centerPosition)
        {
            int windowSize = 5;
            int halfWindow = windowSize / 2;
            int startPos = Math.Max(0, centerPosition - halfWindow);
            int endPos = Math.Min(dockCountPerMove.Count - 1, centerPosition + halfWindow);

            double sum = 0.0;
            int count = 0;
            for (int i = startPos; i <= endPos; i++)
            {
                sum += dockCountPerMove[i];
                count++;
            }

            return count > 0 ? sum / count : 0.0;
        }

        /// <summary>
        /// 计算当前场面的最小cost和拥有最小cost的组合数量（基于已有分析结果）
        /// </summary>
        private static (int minCost, int optionsCount) CalculateMinCostAndOptionsFromExisting(VirtualBattleAnalyzer analyzer)
        {
            var allBestMatches = analyzer.GetAllBestMatchGroups();
            if (allBestMatches.Count == 0)
                return (-1, 0);

            int minCost = allBestMatches.Values.Min(m => m.totalCost);
            int optionsCount = allBestMatches.Values.Count(m => m.totalCost == minCost);

            return (minCost, optionsCount);
        }

        /// <summary>
        /// 正确更新瓦片状态 - 完全复现TileMatchBattle.UpdateTilesState逻辑
        /// </summary>
        private static void UpdateTilesStateCorrectly(Dictionary<int, Tile> allTiles)
        {
            foreach (var tile in allTiles.Values)
            {
                tile.runtimeDependencies = new HashSet<int>();

                if (tile.PileType == PileType.Desk)
                {
                    // 只考虑仍在Desk中的依赖
                    foreach (var dep in tile.Dependencies)
                    {
                        if (allTiles.ContainsKey(dep) && allTiles[dep].PileType == PileType.Desk)
                        {
                            tile.runtimeDependencies.Add(dep);
                        }
                    }

                    // 设置或移除Clickable状态
                    if (tile.runtimeDependencies.Count == 0)
                    {
                        if (!tile.IsSetFlag(ETileFlag.Clickable))
                        {
                            tile.SetFlag(ETileFlag.Clickable);
                        }
                    }
                    else
                    {
                        tile.RemoveFlag(ETileFlag.Clickable);
                    }
                }
            }
        }

        /// <summary>
        /// 使用真实AutoPlay逻辑选择最佳瓦片 - 完全复现BaseCost策略
        /// </summary>
        private static Tile SelectBestTileUsingRealAutoPlayLogic(List<Tile> clickableTiles, VirtualBattleAnalyzer analyzer)
        {
            // 获取所有花色的最佳匹配组合
            var allBestMatches = analyzer.GetAllBestMatchGroups();
            if (allBestMatches.Count == 0)
            {
                return null;
            }

            // 找到Cost最小的组合
            var bestMatch = allBestMatches.Values.OrderBy(m => m.totalCost).First();

            // 在该组合的path中找到可以点击的Tile
            foreach (var id in bestMatch.path)
            {
                if (analyzer.AllTiles.TryGetValue(id, out Tile tile) && clickableTiles.Contains(tile))
                {
                    return tile;
                }
            }

            // 如果path中没有可点击的瓦片，返回null
            return null;
        }

        /// <summary>
        /// 虚拟战斗分析器 - 完全复现TileMatchBattleAnalyzerMgr的行为
        /// </summary>
        private class VirtualBattleAnalyzer
        {
            public Dictionary<int, Tile> AllTiles { get; private set; }
            public List<int> ElementValues => _elementValues;
            private Dictionary<int, List<TileMatchBattleAnalyzerMgr.MatchGroup>> _matchGroups;
            private List<int> _elementValues;

            public VirtualBattleAnalyzer(Dictionary<int, Tile> allTiles, List<int> elementValues)
            {
                AllTiles = allTiles;
                _elementValues = elementValues;
                SetElementValues(elementValues);
            }

            public void SetElementValues(List<int> elementValues)
            {
                _matchGroups = new Dictionary<int, List<TileMatchBattleAnalyzerMgr.MatchGroup>>();
                foreach (var elementValue in elementValues)
                {
                    _matchGroups.Add(elementValue, null);
                }
            }

            /// <summary>
            /// 分析当前对局的所有可消除组合 - 完全复现AnalyzerMgr.Analyze()
            /// </summary>
            public void Analyze()
            {
                if (_matchGroups == null)
                    return;

                // 清空之前的分析结果
                foreach (var key in _matchGroups.Keys.ToList())
                {
                    _matchGroups[key] = new List<TileMatchBattleAnalyzerMgr.MatchGroup>();
                }

                // 分析每个花色的可消除组合
                foreach (var elementValue in _matchGroups.Keys.ToList())
                {
                    GetAllMatches(elementValue);
                }
            }

            /// <summary>
            /// 获取指定花色的所有可消除组合 - 完全复现_getAllMatchs逻辑
            /// </summary>
            private void GetAllMatches(int elementValue)
            {
                // 获取该花色的所有Tile（排除已销毁的Tile）
                List<Tile> allElementValueTiles = AllTiles.Values
                    .Where(t => t.ElementValue == elementValue && !t.IsSetFlag(ETileFlag.Destroyed))
                    .ToList();

                if (allElementValueTiles.Count < 3)
                    return;

                // 按深度排序（runtimeDependencies + 1，Dock区域的Tile深度为0）
                allElementValueTiles.Sort((a, b) =>
                {
                    int depthA = a.PileType == PileType.Dock ? 0 : a.runtimeDependencies.Count + 1;
                    int depthB = b.PileType == PileType.Dock ? 0 : b.runtimeDependencies.Count + 1;
                    return depthA.CompareTo(depthB);
                });

                // 生成所有可能的3个Tile组合
                List<TileMatchBattleAnalyzerMgr.MatchGroup> matchGroups = new List<TileMatchBattleAnalyzerMgr.MatchGroup>();

                for (int i = 0; i < allElementValueTiles.Count - 2; i++)
                {
                    for (int j = i + 1; j < allElementValueTiles.Count - 1; j++)
                    {
                        for (int k = j + 1; k < allElementValueTiles.Count; k++)
                        {
                            List<Tile> matchTiles = new List<Tile>
                            {
                                allElementValueTiles[i],
                                allElementValueTiles[j],
                                allElementValueTiles[k]
                            };

                            int cost = CalculateCost(matchTiles, out HashSet<int> path);

                            var matchGroup = new TileMatchBattleAnalyzerMgr.MatchGroup
                            {
                                matchTiles = matchTiles,
                                totalCost = cost,
                                path = path
                            };

                            matchGroups.Add(matchGroup);
                        }
                    }
                }

                // 按Cost排序，Cost越小越优先
                matchGroups.Sort((a, b) => a.totalCost.CompareTo(b.totalCost));

                _matchGroups[elementValue] = matchGroups;
            }

            /// <summary>
            /// 计算消除成本 - 完全复现_calculateCost逻辑
            /// </summary>
            private int CalculateCost(List<Tile> matchTiles, out HashSet<int> path)
            {
                HashSet<int> allDependencies = new HashSet<int>();
                int totalCost = 0;

                foreach (var tile in matchTiles)
                {
                    if (tile.PileType == PileType.Dock) // 已经在Dock区域的，不需要Cost
                        continue;

                    // 递归收集所有依赖的Tile ID
                    CollectAllDependencies(tile, allDependencies);
                    allDependencies.Add(tile.ID);
                }

                totalCost += allDependencies.Count;
                path = allDependencies;
                return totalCost;
            }

            /// <summary>
            /// 递归收集指定Tile的所有依赖Tile ID - 完全复现_collectAllDependencies逻辑
            /// </summary>
            private void CollectAllDependencies(Tile tile, HashSet<int> allDependencies)
            {
                if (tile == null || tile.runtimeDependencies == null || tile.runtimeDependencies.Count == 0)
                    return;

                foreach (var depId in tile.runtimeDependencies)
                {
                    // 如果这个依赖ID还没有被添加过
                    if (allDependencies.Add(depId))
                    {
                        // 获取依赖的Tile对象
                        if (AllTiles.TryGetValue(depId, out Tile depTile))
                        {
                            // 递归收集这个依赖Tile的依赖
                            CollectAllDependencies(depTile, allDependencies);
                        }
                    }
                }
            }

            /// <summary>
            /// 获取指定花色的最佳匹配组合 - 完全复现GetBestMatchGroup逻辑
            /// </summary>
            public TileMatchBattleAnalyzerMgr.MatchGroup GetBestMatchGroup(int elementValue)
            {
                var matchGroups = GetMatchGroups(elementValue);
                return matchGroups.Count > 0 ? matchGroups[0] : null;
            }

            /// <summary>
            /// 获取指定花色的所有可消除组合 - 完全复现GetMatchGroups逻辑
            /// </summary>
            public List<TileMatchBattleAnalyzerMgr.MatchGroup> GetMatchGroups(int elementValue)
            {
                if (_matchGroups == null || !_matchGroups.ContainsKey(elementValue))
                    return new List<TileMatchBattleAnalyzerMgr.MatchGroup>();

                return _matchGroups[elementValue] ?? new List<TileMatchBattleAnalyzerMgr.MatchGroup>();
            }

            /// <summary>
            /// 获取所有花色的最佳匹配组合 - 完全复现GetAllBestMatchGroups逻辑
            /// </summary>
            public Dictionary<int, TileMatchBattleAnalyzerMgr.MatchGroup> GetAllBestMatchGroups()
            {
                Dictionary<int, TileMatchBattleAnalyzerMgr.MatchGroup> result = new Dictionary<int, TileMatchBattleAnalyzerMgr.MatchGroup>();

                if (_matchGroups == null)
                    return result;

                foreach (var kvp in _matchGroups)
                {
                    var bestMatch = GetBestMatchGroup(kvp.Key);
                    if (bestMatch != null)
                    {
                        result[kvp.Key] = bestMatch;
                    }
                }

                return result;
            }
        }

        /// <summary>
        /// 在Dock中查找匹配的瓦片
        /// </summary>
        private static List<Tile> FindMatchInDock(List<Tile> dockTiles)
        {
            var groups = dockTiles.GroupBy(t => t.ElementValue).ToList();
            var matchGroup = groups.FirstOrDefault(g => g.Count() >= 3);
            return matchGroup?.Take(3).ToList();
        }

        /// <summary>
        /// 加载关卡数据
        /// </summary>
        private static LevelData LoadLevelData(string levelName)
        {
            try
            {
                string jsonFileName;
                if (levelName.StartsWith("Level_"))
                {
                    string numberPart = levelName.Substring(6);
                    int levelNumber = int.Parse(numberPart);
                    jsonFileName = $"{100000 + levelNumber}.json";
                }
                else
                {
                    jsonFileName = levelName.EndsWith(".json") ? levelName : $"{levelName}.json";
                }

                string jsonPath = Path.Combine(Application.dataPath, "..", "Tools", "Config", "Json", "Levels", jsonFileName);
                jsonPath = Path.GetFullPath(jsonPath);

                if (!File.Exists(jsonPath))
                {
                    Debug.LogError($"关卡JSON文件不存在: {jsonPath}");
                    return null;
                }

                string jsonContent = File.ReadAllText(jsonPath);
                return JsonUtility.FromJson<LevelData>(jsonContent);
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
        /// 创建可用花色列表
        /// </summary>
        private static List<int> CreateAvailableColors(int colorCount)
        {
            var standardColors = new int[] { 101, 102, 103, 201, 202, 301, 302, 401, 402, 403, 501, 502, 601, 602, 701, 702, 703, 801, 802 };

            if (colorCount <= standardColors.Length)
            {
                var shuffled = new List<int>(standardColors);
                for (int i = 0; i < shuffled.Count; i++)
                {
                    int randomIndex = UnityEngine.Random.Range(i, shuffled.Count);
                    (shuffled[i], shuffled[randomIndex]) = (shuffled[randomIndex], shuffled[i]);
                }
                return shuffled.GetRange(0, colorCount);
            }
            else
            {
                return new List<int>(standardColors);
            }
        }

        /// <summary>
        /// 计算关卡总瓦片数量
        /// </summary>
        private static int CalculateTotalTileCount(LevelData levelData)
        {
            return levelData.Layers.Sum(layer => layer.tiles.Length);
        }

        /// <summary>
        /// 批量运行分析 - 支持-1枚举值的多配置组合
        /// </summary>
        public static List<AnalysisResult> RunBatchAnalysis(RunConfig config)
        {
            CsvConfigManager.LoadCsvConfigs();
            var results = new List<AnalysisResult>();

            // 应用种子配置
            if (config.UseFixedSeed)
            {
                UnityEngine.Random.InitState(config.FixedSeedValue);
                Debug.Log($"使用固定随机种子: {config.FixedSeedValue} (确保结果可重现)");
            }
            else
            {
                Debug.Log("使用随机种子模式 (每次结果不同)");
            }

            Debug.Log($"开始批量分析: {config.GetConfigDescription()}");

            // 预计算总任务数
            int totalTasks = 0;
            var levelConfigs = new Dictionary<string, (int[][] experienceModes, int[] colorCounts)>();

            for (int i = 1; i <= config.TestLevelCount; i++)
            {
                string levelName = $"Level_{i:D3}";
                var experienceModes = CsvConfigManager.ResolveExperienceModes(config.ExperienceConfigEnum, i);
                var colorCounts = CsvConfigManager.ResolveColorCounts(config.ColorCountConfigEnum, i);
                levelConfigs[levelName] = (experienceModes, colorCounts);
                totalTasks += experienceModes.Length * colorCounts.Length * config.RunsPerLevel;
            }

            Debug.Log($"关卡数量: {config.TestLevelCount}, 总任务数: {totalTasks}");

            int completedTasks = 0;
            int uniqueIdCounter = 1; // 唯一ID计数器

            foreach (var kvp in levelConfigs)
            {
                string levelName = kvp.Key;
                var (experienceModes, colorCounts) = kvp.Value;
                int terrainId = int.Parse(levelName.Substring(6)); // 提取Level_001中的001

                foreach (var experienceMode in experienceModes)
                {
                    foreach (var colorCount in colorCounts)
                    {
                        // 根据配置生成多次运行
                        for (int runIndex = 0; runIndex < config.RunsPerLevel; runIndex++)
                        {
                            int randomSeed = config.GetSeedForRun(terrainId, runIndex);

                            completedTasks++;
                            Debug.Log($"[{completedTasks}/{totalTasks}] 分析关卡 {levelName}: " +
                                     $"体验[{string.Join(",", experienceMode)}], 花色{colorCount}, 种子{randomSeed}");

                            var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                            result.TerrainId = terrainId;
                            result.UniqueId = $"BA_{uniqueIdCounter:D6}"; // 生成唯一ID：BA_000001, BA_000002...
                            uniqueIdCounter++;
                            results.Add(result);
                        }
                    }
                }
            }

            Debug.Log($"批量分析完成: {results.Count} 个任务结果");
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

                // CSV表头 - 添加InitialMinCost字段
                csv.AppendLine("UniqueId,TerrainId,LevelName,AlgorithmName,ExperienceMode,ColorCount,TotalTiles,RandomSeed," +
                              "GameCompleted,TotalMoves,GameDurationMs,CompletionStatus," +
                              "TotalAnalysisTimeMs,SuccessfulGroups,InitialMinCost," +
                              "DifficultyPosition,TileIdSequence,DockCountPerMove,PeakDockCount,DockAfterTrioMatch,SafeOptionCounts," +
                              "MinCostAfterTrioMatch,MinCostOptionsAfterTrioMatch,ErrorMessage");

                foreach (var result in results)
                {
                    string expMode = $"[{string.Join(",", result.ExperienceMode)}]";
                    string tileSequence = result.TileIdSequence.Count > 0 ? string.Join(",", result.TileIdSequence) : "";
                    string dockCounts = result.DockCountPerMove.Count > 0 ? string.Join(",", result.DockCountPerMove) : "";
                    string dockAfterTrio = result.DockAfterTrioMatch.Count > 0 ? string.Join(",", result.DockAfterTrioMatch) : "";
                    string safeOptions = result.SafeOptionCounts.Count > 0 ? string.Join(",", result.SafeOptionCounts) : "";
                    string minCostAfterTrio = result.MinCostAfterTrioMatch.Count > 0 ? string.Join(",", result.MinCostAfterTrioMatch) : "";
                    string minCostOptionsAfterTrio = result.MinCostOptionsAfterTrioMatch.Count > 0 ? string.Join(",", result.MinCostOptionsAfterTrioMatch) : "";

                    csv.AppendLine($"{result.UniqueId},{result.TerrainId},{result.LevelName},{result.AlgorithmName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.RandomSeed}," +
                                  $"{result.GameCompleted},{result.TotalMoves},{result.GameDurationMs},\"{result.CompletionStatus}\"," +
                                  $"{result.TotalAnalysisTimeMs},{result.SuccessfulGroups},{result.InitialMinCost}," +
                                  $"{result.DifficultyPosition:F4},\"{tileSequence}\",\"{dockCounts}\",{result.PeakDockCount},\"{dockAfterTrio}\",\"{safeOptions}\"," +
                                  $"\"{minCostAfterTrio}\",\"{minCostOptionsAfterTrio}\",\"{result.ErrorMessage ?? ""}\"");
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
        /// Unity Editor菜单：运行BattleAnalyzer批量分析（使用默认配置）
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/运行批量分析")]
        public static void RunBatchAnalysisFromMenu()
        {
            var config = new RunConfig(); // 使用完全默认的配置
            config.OutputDirectory = Path.Combine(Application.dataPath, "_Editor/BattleAnalysisResults");

            Debug.Log($"=== 开始批量分析（默认配置） ===");
            Debug.Log($"配置详情: {config.GetConfigDescription()}");
            Debug.Log($"关卡数量: {config.TestLevelCount}");

            var results = RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string seedSuffix = config.UseFixedSeed ? $"_Fixed{config.FixedSeedValue}" : "_Random";
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis{seedSuffix}_{timestamp}.csv");

            ExportToCsv(results, csvPath);

            Debug.Log($"批量分析完成! 成功分析 {results.Count} 个任务");
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹
            if (Directory.Exists(config.OutputDirectory))
            {
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
            }
        }
#endif
    }
}