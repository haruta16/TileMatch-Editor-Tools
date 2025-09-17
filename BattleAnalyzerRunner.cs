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
        /// CSV配置行数据
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
            public int TerrainId { get; set; }
            public string LevelName { get; set; }
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
            public int TotalAnalysisCalls { get; set; }
            public int TotalAnalysisTimeMs { get; set; }
            public int SuccessfulMoves { get; set; }
            public List<int> TileIdSequence { get; set; } = new List<int>();
            public List<int> DockCountPerMove { get; set; } = new List<int>();

            // 关键快照数据
            public int PeakDockCount { get; set; }
            public int MinMovesToComplete { get; set; }

            public string ErrorMessage { get; set; }
        }

        /// <summary>
        /// 批量运行配置
        /// </summary>
        public class RunConfig
        {
            public int TestLevelCount = 50;  // 测试关卡数量
            public int ExperienceConfigType = 1; // 1=exp-fix-1, 2=exp-fix-2
            public int ColorCountConfigType = 1; // 1=type-count-1, 2=type-count-2
            public string OutputDirectory = "BattleAnalysisResults";
        }

        private static Dictionary<int, CsvLevelConfig> _csvConfigs = null;

        /// <summary>
        /// 加载CSV配置
        /// </summary>
        private static void LoadCsvConfigs()
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
                    var parts = ParseCsvLine(lines[i]);
                    if (parts.Length >= 7 && int.TryParse(parts[0], out int terrainId))
                    {
                        var config = new CsvLevelConfig
                        {
                            TerrainId = terrainId,
                            ExpFix1 = ParseIntArray(parts[1]),
                            ExpFix2 = ParseIntArray(parts[2]),
                            ExpRange1 = ParseIntArray(parts[3]),
                            TypeCount1 = ParseIntOrDefault(parts[4], 7),
                            TypeCount2 = ParseIntOrDefault(parts[5], 8),
                            TypeRange1 = ParseIntOrDefault(parts[6], 7)
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
        /// 解析CSV行
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
        /// 解析整数数组
        /// </summary>
        private static int[] ParseIntArray(string arrayStr)
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
        private static int ParseIntOrDefault(string str, int defaultValue)
        {
            return int.TryParse(str.Trim(), out int result) ? result : defaultValue;
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
            int analysisCalls = 0;
            int analysisTimeMs = 0;
            int successfulMoves = 0;
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
                analysisCalls++;
                virtualAnalyzer.Analyze();
                analysisStopwatch.Stop();
                analysisTimeMs += (int)analysisStopwatch.ElapsedMilliseconds;

                // 3. 获取可点击的瓦片
                var clickableTiles = allTiles.Values
                    .Where(t => t.PileType == PileType.Desk && t.IsClickable)
                    .ToList();

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
                    successfulMoves++;
                    matchOccurred = true;
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
            result.TotalAnalysisCalls = analysisCalls;
            result.TotalAnalysisTimeMs = analysisTimeMs;
            result.SuccessfulMoves = successfulMoves;
            result.PeakDockCount = peakDockCount;
            result.MinMovesToComplete = result.TileIdSequence.Count;

            if (string.IsNullOrEmpty(result.CompletionStatus))
            {
                result.CompletionStatus = "Incomplete";
            }

            return result;
        }

        /// <summary>
        /// 正确更新瓦片状态 - 完全复现TileMatchBattle.UpdateTilesState逻辑
        /// </summary>
        private static void UpdateTilesStateCorrectly(Dictionary<int, Tile> allTiles)
        {
            foreach (var tile in allTiles.Values)
            {
                tile.runtimeDependencies = new List<int>();

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
        /// 批量运行分析
        /// </summary>
        public static List<AnalysisResult> RunBatchAnalysis(RunConfig config)
        {
            LoadCsvConfigs();

            var results = new List<AnalysisResult>();

            // 暂时注释固定种子，使用随机种子
            // UnityEngine.Random.InitState(12345678);

            for (int i = 1; i <= config.TestLevelCount; i++)
            {
                string levelName = $"Level_{i:D3}";

                if (!_csvConfigs.TryGetValue(i, out var csvConfig))
                {
                    Debug.LogWarning($"未找到地形ID {i} 的配置，跳过");
                    continue;
                }

                // 根据配置类型选择体验模式和花色数量
                int[] experienceMode = config.ExperienceConfigType == 1 ? csvConfig.ExpFix1 : csvConfig.ExpFix2;
                int colorCount = config.ColorCountConfigType == 1 ? csvConfig.TypeCount1 : csvConfig.TypeCount2;

                // 为每个关卡生成3次不同的随机配置
                for (int runIndex = 0; runIndex < 3; runIndex++)
                {
                    // 生成随机种子
                    int randomSeed = UnityEngine.Random.Range(1, int.MaxValue);

                    Debug.Log($"正在分析关卡 {levelName} (第{runIndex + 1}次), 体验模式: [{string.Join(",", experienceMode)}], 花色数量: {colorCount}, 随机种子: {randomSeed}");

                    var result = RunSingleLevelAnalysis(levelName, experienceMode, colorCount, randomSeed);
                    result.TerrainId = i;
                    results.Add(result);
                }
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

                // CSV表头 - 添加RandomSeed字段
                csv.AppendLine("TerrainId,LevelName,ExperienceMode,ColorCount,TotalTiles,RandomSeed," +
                              "GameCompleted,TotalMoves,GameDurationMs,CompletionStatus," +
                              "TotalAnalysisCalls,TotalAnalysisTimeMs,SuccessfulMoves," +
                              "TileIdSequence,DockCountPerMove,PeakDockCount,MinMovesToComplete,ErrorMessage");

                foreach (var result in results)
                {
                    string expMode = $"[{string.Join(",", result.ExperienceMode)}]";
                    string tileSequence = result.TileIdSequence.Count > 0 ? string.Join(",", result.TileIdSequence) : "";
                    string dockCounts = result.DockCountPerMove.Count > 0 ? string.Join(",", result.DockCountPerMove) : "";

                    csv.AppendLine($"{result.TerrainId},{result.LevelName},\"{expMode}\",{result.ColorCount},{result.TotalTiles},{result.RandomSeed}," +
                                  $"{result.GameCompleted},{result.TotalMoves},{result.GameDurationMs},\"{result.CompletionStatus}\"," +
                                  $"{result.TotalAnalysisCalls},{result.TotalAnalysisTimeMs},{result.SuccessfulMoves}," +
                                  $"\"{tileSequence}\",\"{dockCounts}\",{result.PeakDockCount},{result.MinMovesToComplete},\"{result.ErrorMessage ?? ""}\"");
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
        /// Unity Editor菜单：运行BattleAnalyzer批量分析
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/运行批量分析")]
        public static void RunBatchAnalysisFromMenu()
        {
            var config = new RunConfig
            {
                TestLevelCount = 50,
                ExperienceConfigType = 1,
                ColorCountConfigType = 1,
                OutputDirectory = Path.Combine(Application.dataPath, "_Editor/BattleAnalysisResults")
            };

            Debug.Log($"开始批量BattleAnalyzer分析，关卡数量: {config.TestLevelCount}");

            var results = RunBatchAnalysis(config);

            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var csvPath = Path.Combine(config.OutputDirectory, $"BattleAnalysis_{timestamp}.csv");

            ExportToCsv(results, csvPath);

            Debug.Log($"批量分析完成! 成功分析 {results.Count} 个关卡");
            Debug.Log($"结果已保存到: {csvPath}");

            // 打开输出文件夹
            if (Directory.Exists(config.OutputDirectory))
            {
                System.Diagnostics.Process.Start("explorer.exe", config.OutputDirectory.Replace('/', '\\'));
            }
        }

        /// <summary>
        /// Unity Editor菜单：测试单个关卡分析
        /// </summary>
        [MenuItem("TileMatch/BattleAnalyzer/测试单个关卡")]
        public static void TestSingleLevel()
        {
            // 生成随机种子进行测试
            int randomSeed = UnityEngine.Random.Range(1, int.MaxValue);
            var result = RunSingleLevelAnalysis("Level_001", new int[] { 1, 2, 3 }, 7, randomSeed);

            Debug.Log($"单关卡测试结果:");
            Debug.Log($"关卡: {result.LevelName}");
            Debug.Log($"随机种子: {result.RandomSeed}");
            Debug.Log($"游戏完成: {result.GameCompleted}");
            Debug.Log($"总移动数: {result.TotalMoves}");
            Debug.Log($"游戏时长: {result.GameDurationMs}ms");
            Debug.Log($"分析调用次数: {result.TotalAnalysisCalls}");
            Debug.Log($"成功消除次数: {result.SuccessfulMoves}");
            Debug.Log($"峰值Dock数量: {result.PeakDockCount}");

            if (!string.IsNullOrEmpty(result.ErrorMessage))
            {
                Debug.LogError($"错误: {result.ErrorMessage}");
            }
        }
#endif
    }
}