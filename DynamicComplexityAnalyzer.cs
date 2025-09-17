using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using DGuo.Client.TileMatch;
using DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo;

namespace DGuo.Client.TileMatch.DesignerAlgo.Evaluation
{
    /// <summary>
    /// 动态复杂度分析器 - 通过自动游玩获取关卡动态复杂度指标
    /// 架构设计：纯动态分析，复用现有AutoPlayStrategy规则部分，确保游戏规则100%一致性
    /// </summary>
    public class DynamicComplexityAnalyzer
    {
        #region 数据结构定义

        /// <summary>
        /// 动态复杂度指标数据结构 - 可扩展的指标容器
        /// </summary>
        [System.Serializable]
        public class DynamicComplexityMetrics
        {
            // 基础游戏流程指标
            public int TotalMoves { get; set; } = 0;                    // 总移动步数
            public int GameDurationMs { get; set; } = 0;                // 游戏时长(毫秒)
            public bool GameCompleted { get; set; } = false;            // 游戏是否完成
            public string CompletionStatus { get; set; } = "Unknown";   // 完成状态 ("Victory"/"Defeat"/"Error")

            // 扩展指标字段 - 为后续具体指标设计预留
            public Dictionary<string, object> ExtendedMetrics { get; set; } = new Dictionary<string, object>();
            public List<int> OptimalMoveTileIds { get; set; } = new List<int>();
            public List<int> DockCountPerMove { get; set; } = new List<int>();
            public int PeakDockDuringSolution { get; set; } = 0;

            /// <summary>
            /// 添加扩展指标 - 支持任意类型的动态指标添加
            /// </summary>
            public void AddMetric<T>(string key, T value) where T : struct
            {
                ExtendedMetrics[key] = value;
            }

            /// <summary>
            /// 获取扩展指标 - 类型安全的指标获取
            /// </summary>
            public T GetMetric<T>(string key, T defaultValue = default) where T : struct
            {
                return ExtendedMetrics.TryGetValue(key, out var value) && value is T ? (T)value : defaultValue;
            }
        }

        #endregion

        #region 核心接口定义

        /// <summary>
        /// 游戏规则引擎接口 - 抽象游戏核心逻辑，便于复用和测试
        /// </summary>
        public interface IGameRulesEngine
        {
            /// <summary>
            /// 初始化游戏状态
            /// </summary>
            void InitializeGame(List<Tile> tiles);

            /// <summary>
            /// 获取当前可点击的瓦片列表
            /// </summary>
            List<Tile> GetClickableTiles();

            /// <summary>
            /// 执行瓦片点击操作
            /// </summary>
            bool ExecuteTileClick(Tile tile);

            /// <summary>
            /// 检查游戏是否结束
            /// </summary>
            bool IsGameOver();

            /// <summary>
            /// 检查游戏是否获胜
            /// </summary>
            bool IsGameWon();

            /// <summary>
            /// 获取当前游戏状态快照 - 为动态指标收集提供数据源
            /// </summary>
            GameStateSnapshot GetCurrentState();
        }

        /// <summary>
        /// 游戏状态快照 - 为动态指标计算提供数据源
        /// </summary>
        [System.Serializable]
        public class GameStateSnapshot
        {
            public List<Tile> DeskTiles { get; set; } = new List<Tile>();     // 当前桌面瓦片
            public List<Tile> DockTiles { get; set; } = new List<Tile>();     // 当前Dock瓦片
            public int MoveCount { get; set; } = 0;                           // 当前移动计数
            public DateTime Timestamp { get; set; } = DateTime.Now;           // 状态时间戳

            // 扩展状态数据 - 为具体指标计算预留
            public Dictionary<string, object> StateData { get; set; } = new Dictionary<string, object>();
        }

        /// <summary>
        /// 自动游玩策略接口 - 抽象决策逻辑，支持多种分析策略
        /// </summary>
        public interface IAutoPlayStrategy
        {
            /// <summary>
            /// 策略名称
            /// </summary>
            string StrategyName { get; }

            /// <summary>
            /// 根据当前状态选择下一步操作的瓦片
            /// </summary>
            Tile SelectNextTile(List<Tile> clickableTiles, GameStateSnapshot currentState);

            /// <summary>
            /// 策略初始化 - 在游戏开始前调用
            /// </summary>
            void Initialize(List<Tile> allTiles);

            /// <summary>
            /// 策略重置 - 为下一轮游戏准备
            /// </summary>
            void Reset();
        }

        /// <summary>
        /// 动态指标计算器接口 - 抽象指标计算逻辑，支持扩展
        /// </summary>
        public interface IDynamicMetricsCalculator
        {
            /// <summary>
            /// 处理游戏状态变化 - 在每次移动后调用
            /// </summary>
            void ProcessStateChange(GameStateSnapshot previousState, GameStateSnapshot currentState);

            /// <summary>
            /// 计算最终指标 - 在游戏结束后调用
            /// </summary>
            void CalculateFinalMetrics(DynamicComplexityMetrics metrics);

            /// <summary>
            /// 重置计算器状态
            /// </summary>
            void Reset();
        }

        #endregion

        #region 默认实现类

        /// <summary>
        /// 基于现有AutoPlayStrategy的游戏规则引擎实现
        /// </summary>
        private class AutoPlayBasedGameEngine : IGameRulesEngine
        {
            private List<Tile> deskTiles;
            private List<Tile> dockTiles;
            private int moveCount;

            public void InitializeGame(List<Tile> tiles)
            {
                deskTiles = new List<Tile>(tiles.Where(t => t.PileType == PileType.Desk));
                dockTiles = new List<Tile>();
                moveCount = 0;

                // 更新初始可点击状态
                UpdateTileStates();
            }

            public List<Tile> GetClickableTiles()
            {
                // 复用现有的可点击逻辑 - 确保规则一致性
                return deskTiles.Where(t => IsClickable(t)).ToList();
            }

            public bool ExecuteTileClick(Tile tile)
            {
                if (!IsClickable(tile)) return false;

                // 模拟瓦片移动到Dock的逻辑
                deskTiles.Remove(tile);
                dockTiles.Add(tile);
                moveCount++;

                // 检查并处理三消逻辑（复用现有Rule.CheckDockMatch逻辑）
                ProcessDockMatches();

                // 更新瓦片可点击状态
                UpdateTileStates();

                return true;
            }

            public bool IsGameOver()
            {
                return IsGameWon() || dockTiles.Count >= 7;
            }

            public bool IsGameWon()
            {
                return deskTiles.Count == 0;
            }

            public GameStateSnapshot GetCurrentState()
            {
                return new GameStateSnapshot
                {
                    DeskTiles = new List<Tile>(deskTiles),
                    DockTiles = new List<Tile>(dockTiles),
                    MoveCount = moveCount,
                    Timestamp = DateTime.Now
                };
            }

            private bool IsClickable(Tile tile)
            {
                // 检查依赖关系：所有依赖的瓦片都已被移除
                if (tile.Dependencies == null) return true;

                foreach (var depId in tile.Dependencies)
                {
                    // 如果依赖的瓦片还在桌面上，则不可点击
                    if (deskTiles.Any(t => t.ID == depId))
                    {
                        return false;
                    }
                }
                return true;
            }

            private void ProcessDockMatches()
            {
                // 复用现有的三消检测和消除逻辑
                var groups = dockTiles.GroupBy(t => t.ElementValue).Where(g => g.Count() >= 3);
                foreach (var group in groups)
                {
                    var tilesToRemove = group.Take(3).ToList();
                    foreach (var tile in tilesToRemove)
                    {
                        dockTiles.Remove(tile);
                    }
                }
            }

            private void UpdateTileStates()
            {
                // 更新所有桌面瓦片的可点击状态
                foreach (var tile in deskTiles)
                {
                    // 可点击状态已通过IsClickable方法动态计算
                    // 这里可以添加额外的状态更新逻辑
                }
            }
        }

        /// <summary>
        /// 默认自动游玩策略实现 - 最简单的贪心策略
        /// </summary>
        private class DefaultAutoPlayStrategy : IAutoPlayStrategy
        {
            public string StrategyName => "Default-Greedy";

            private List<Tile> allTiles;

            public void Initialize(List<Tile> tiles)
            {
                allTiles = tiles;
            }

            public Tile SelectNextTile(List<Tile> clickableTiles, GameStateSnapshot currentState)
            {
                // 最简单的选择逻辑：选择第一个可用瓦片
                // 具体策略逻辑将在后续需求中实现
                return clickableTiles.FirstOrDefault();
            }

            public void Reset()
            {
                allTiles = null;
            }
        }

        /// <summary>
        /// 基于TileMatchBattleAnalyzerMgr算法的自动游玩策略
        /// 复用现有算法进行贪心选择，用于对比测试
        /// </summary>
        private class BattleAnalyzerAutoPlayStrategy : IAutoPlayStrategy
        {
            public string StrategyName => "BattleAnalyzer-Greedy";

            private Dictionary<int, List<MatchGroup>> matchGroups;
            private Dictionary<int, TileInfo> tileMap;
            private List<TileInfo> allTileInfos;

            // 公开的统计数据
            public long AnalysisTimeMs { get; private set; } = 0;
            public int TotalAnalysisCalls { get; private set; } = 0;
            public int SuccessfulMoves { get; private set; } = 0;

            private class TileInfo
            {
                public int Id;
                public int Element;
                public int[] Deps;
                public HashSet<int> RuntimeDependencies; // 运行时依赖，动态更新
                public PileType PileType = PileType.Desk;
                public bool IsDestroyed = false;
            }

            private class MatchGroup
            {
                public List<TileInfo> matchTiles;
                public int totalCost;
                public HashSet<int> path;
            }

            public void Initialize(List<Tile> allTiles)
            {
                tileMap = new Dictionary<int, TileInfo>(allTiles.Count);
                allTileInfos = new List<TileInfo>(allTiles.Count);

                foreach (var t in allTiles)
                {
                    var info = new TileInfo
                    {
                        Id = t.ID,
                        Element = t.ElementValue,
                        Deps = t.Dependencies ?? System.Array.Empty<int>(),
                        RuntimeDependencies = new HashSet<int>(t.Dependencies ?? System.Array.Empty<int>()),
                        PileType = t.PileType,
                        IsDestroyed = false
                    };
                    tileMap[t.ID] = info;
                    allTileInfos.Add(info);
                }
            }

            public Tile SelectNextTile(List<Tile> clickableTiles, GameStateSnapshot currentState)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                TotalAnalysisCalls++;

                try
                {
                    // 更新瓦片状态
                    UpdateTileStates(currentState);

                    // 执行分析
                    AnalyzeCurrentState();

                    // 选择最佳移动
                    var bestTile = SelectBestMove(clickableTiles);

                    if (bestTile != null)
                        SuccessfulMoves++;

                    return bestTile;
                }
                finally
                {
                    sw.Stop();
                    AnalysisTimeMs += sw.ElapsedMilliseconds;
                }
            }

            private void UpdateTileStates(GameStateSnapshot currentState)
            {
                // 重置所有瓦片状态
                foreach (var info in allTileInfos)
                {
                    info.PileType = PileType.Desk;
                    info.IsDestroyed = false;
                    // 重置运行时依赖为初始依赖
                    info.RuntimeDependencies = new HashSet<int>(info.Deps);
                }

                // 更新Dock中的瓦片
                if (currentState.DockTiles != null)
                {
                    foreach (var dockTile in currentState.DockTiles)
                    {
                        if (tileMap.TryGetValue(dockTile.ID, out var info))
                        {
                            info.PileType = PileType.Dock;
                        }
                    }
                }

                // 标记已销毁的瓦片（不在Desk也不在Dock中的）
                var deskIds = currentState.DeskTiles?.Select(t => t.ID).ToHashSet() ?? new HashSet<int>();
                var dockIds = currentState.DockTiles?.Select(t => t.ID).ToHashSet() ?? new HashSet<int>();
                var destroyedIds = new HashSet<int>();

                foreach (var info in allTileInfos)
                {
                    if (!deskIds.Contains(info.Id) && !dockIds.Contains(info.Id))
                    {
                        info.IsDestroyed = true;
                        destroyedIds.Add(info.Id);
                    }
                }

                // 🆕 关键修复：更新运行时依赖关系 - 移除已销毁的瓦片依赖
                foreach (var info in allTileInfos)
                {
                    if (!info.IsDestroyed)
                    {
                        // 从运行时依赖中移除所有已销毁的瓦片ID
                        info.RuntimeDependencies.ExceptWith(destroyedIds);
                    }
                }
            }

            private void AnalyzeCurrentState()
            {
                // 获取所有元素类型
                var elementValues = allTileInfos.Where(t => !t.IsDestroyed)
                    .Select(t => t.Element).Distinct().ToList();

                matchGroups = new Dictionary<int, List<MatchGroup>>();

                foreach (var elementValue in elementValues)
                {
                    matchGroups[elementValue] = new List<MatchGroup>();
                    GetAllMatches(elementValue);
                }
            }

            private void GetAllMatches(int elementValue)
            {
                // 获取该花色的所有未销毁Tile
                var allElementValueTiles = allTileInfos
                    .Where(t => t.Element == elementValue && !t.IsDestroyed)
                    .ToList();

                if (allElementValueTiles.Count < 3)
                    return;

                // 按深度排序（完全匹配 runtimeDependencies.Count + 1 的逻辑）
                allElementValueTiles.Sort((a, b) =>
                {
                    int depthA = a.PileType == PileType.Dock ? 0 : a.RuntimeDependencies.Count + 1;
                    int depthB = b.PileType == PileType.Dock ? 0 : b.RuntimeDependencies.Count + 1;
                    return depthA.CompareTo(depthB);
                });

                // 生成所有可能的3个Tile组合
                var matchGroups = new List<MatchGroup>();

                for (int i = 0; i < allElementValueTiles.Count - 2; i++)
                {
                    for (int j = i + 1; j < allElementValueTiles.Count - 1; j++)
                    {
                        for (int k = j + 1; k < allElementValueTiles.Count; k++)
                        {
                            var matchTiles = new List<TileInfo>
                            {
                                allElementValueTiles[i],
                                allElementValueTiles[j],
                                allElementValueTiles[k]
                            };

                            int cost = CalculateCost(matchTiles, out HashSet<int> path);

                            var matchGroup = new MatchGroup
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
                this.matchGroups[elementValue] = matchGroups;
            }

            private int CalculateCost(List<TileInfo> matchTiles, out HashSet<int> path)
            {
                var allDependencies = new HashSet<int>();
                int totalCost = 0;

                foreach (var tile in matchTiles)
                {
                    if (tile.PileType == PileType.Dock) // 已经在Dock区域的，不需要Cost
                        continue;

                    // 递归收集所有依赖的Tile ID
                    CollectAllDependencies(tile, allDependencies);
                    allDependencies.Add(tile.Id);
                }

                totalCost += allDependencies.Count;
                path = allDependencies;
                return totalCost;
            }

            private void CollectAllDependencies(TileInfo tile, HashSet<int> allDependencies)
            {
                if (tile?.RuntimeDependencies == null || tile.RuntimeDependencies.Count == 0)
                    return;

                foreach (var depId in tile.RuntimeDependencies)
                {
                    // 如果这个依赖ID还没有被添加过
                    if (allDependencies.Add(depId))
                    {
                        // 获取依赖的Tile对象
                        if (tileMap.TryGetValue(depId, out TileInfo depTile) && !depTile.IsDestroyed)
                        {
                            // 递归收集这个依赖Tile的依赖
                            CollectAllDependencies(depTile, allDependencies);
                        }
                    }
                }
            }

            private Tile SelectBestMove(List<Tile> clickableTiles)
            {
                if (matchGroups == null || clickableTiles.Count == 0)
                    return clickableTiles.FirstOrDefault();

                Tile bestTile = null;
                int bestCost = int.MaxValue;
                int bestElementValue = -1;

                // 遍历所有可点击瓦片，找到成本最低的选择
                foreach (var tile in clickableTiles)
                {
                    int elementValue = tile.ElementValue;

                    if (!matchGroups.TryGetValue(elementValue, out var groups) || groups.Count == 0)
                        continue;

                    // 找到包含当前瓦片的最佳组合
                    var bestGroupForTile = groups.FirstOrDefault(g =>
                        g.matchTiles.Any(mt => mt.Id == tile.ID));

                    if (bestGroupForTile != null && bestGroupForTile.totalCost < bestCost)
                    {
                        bestCost = bestGroupForTile.totalCost;
                        bestTile = tile;
                        bestElementValue = elementValue;
                    }
                }

                // 如果没有找到合适的，就选择第一个可用的
                return bestTile ?? clickableTiles.FirstOrDefault();
            }

            public void Reset()
            {
                matchGroups?.Clear();
                tileMap?.Clear();
                if (allTileInfos != null)
                {
                    foreach (var info in allTileInfos)
                    {
                        info.RuntimeDependencies?.Clear();
                    }
                    allTileInfos.Clear();
                }
                AnalysisTimeMs = 0;
                TotalAnalysisCalls = 0;
                SuccessfulMoves = 0;
            }
        }

        /// <summary>
        /// 默认动态指标计算器 - 基础指标计算
        /// </summary>
        private class DefaultMetricsCalculator : IDynamicMetricsCalculator
        {
            private List<GameStateSnapshot> stateHistory;
            private DateTime gameStartTime;

            public DefaultMetricsCalculator()
            {
                stateHistory = new List<GameStateSnapshot>();
            }

            public void ProcessStateChange(GameStateSnapshot previousState, GameStateSnapshot currentState)
            {
                if (stateHistory.Count == 0)
                {
                    gameStartTime = currentState.Timestamp;
                }
                stateHistory.Add(currentState);
            }

            public void CalculateFinalMetrics(DynamicComplexityMetrics metrics)
            {
                if (stateHistory.Count == 0) return;

                var lastState = stateHistory.Last();
                metrics.TotalMoves = lastState.MoveCount;
                metrics.GameDurationMs = (int)(lastState.Timestamp - gameStartTime).TotalMilliseconds;

                // 具体的复杂度指标计算将在后续需求中实现
                // 这里预留扩展接口
            }

            public void Reset()
            {
                stateHistory.Clear();
            }
        }

        #endregion

        #region 公共API接口

        /// <summary>
        /// 算法类型枚举
        /// </summary>
        public enum AlgorithmType
        {
            OptimalDFS,         // 原有的最优DFS算法
            BattleAnalyzer,     // TileMatchBattleAnalyzerMgr算法
            DefaultGreedy       // 简单贪心算法
        }

        /// <summary>
        /// 算法执行策略枚举
        /// </summary>
        public enum AlgorithmStrategy
        {
            Single,    // 运行第一个算法
            Parallel,  // 并行运行所有算法
            Fallback,  // 依序尝试直到成功
            Best       // 选择最优结果
        }

        /// <summary>
        /// 算法执行结果
        /// </summary>
        [System.Serializable]
        public class ExecutionResult
        {
            public bool Success { get; set; }
            public string ErrorMessage { get; set; }
            public List<AlgorithmResult> Results { get; set; } = new List<AlgorithmResult>();
            public AlgorithmResult BestResult => Results.Where(r => r.Success).OrderBy(r => r.Metrics?.TotalMoves ?? int.MaxValue).FirstOrDefault();
        }

        /// <summary>
        /// 单个算法执行结果
        /// </summary>
        [System.Serializable]
        public class AlgorithmResult
        {
            public AlgorithmType Type { get; set; }
            public string Name { get; set; }
            public DynamicComplexityMetrics Metrics { get; set; }
            public bool Success { get; set; }
            public string Error { get; set; }
        }

        /// <summary>
        /// 算法执行器 - 统一管理多种算法执行策略
        /// </summary>
        public static class AlgorithmExecutor
        {
            /// <summary>
            /// 根据策略执行算法
            /// </summary>
            public static ExecutionResult Execute(
                AlgorithmStrategy strategy,
                AlgorithmType[] algorithmList,
                List<Tile> tiles,
                int[] experienceMode,
                TerrainAnalysisResult terrainAnalysis,
                bool enableRetry = true,
                int maxRetryCount = 2)
            {
                return strategy switch
                {
                    AlgorithmStrategy.Single => ExecuteSingle(algorithmList[0], tiles, experienceMode, terrainAnalysis, enableRetry, maxRetryCount),
                    AlgorithmStrategy.Parallel => ExecuteParallel(algorithmList, tiles, experienceMode, terrainAnalysis),
                    AlgorithmStrategy.Fallback => ExecuteFallback(algorithmList, tiles, experienceMode, terrainAnalysis, maxRetryCount),
                    AlgorithmStrategy.Best => ExecuteBest(algorithmList, tiles, experienceMode, terrainAnalysis),
                    _ => throw new ArgumentException($"未支持的算法策略: {strategy}")
                };
            }

            /// <summary>
            /// 执行单一算法
            /// </summary>
            private static ExecutionResult ExecuteSingle(
                AlgorithmType algorithm,
                List<Tile> tiles,
                int[] experienceMode,
                TerrainAnalysisResult terrainAnalysis,
                bool enableRetry,
                int maxRetryCount)
            {
                var result = new ExecutionResult();
                int attempt = 0;

                while (attempt <= maxRetryCount)
                {
                    try
                    {
                        var metrics = AnalyzeGameplayComplexity(tiles, experienceMode, terrainAnalysis, algorithm);
                        result.Results.Add(new AlgorithmResult
                        {
                            Type = algorithm,
                            Name = algorithm.ToString(),
                            Metrics = metrics,
                            Success = true
                        });
                        result.Success = true;
                        break;
                    }
                    catch (Exception ex)
                    {
                        attempt++;
                        if (!enableRetry || attempt > maxRetryCount)
                        {
                            result.Results.Add(new AlgorithmResult
                            {
                                Type = algorithm,
                                Name = algorithm.ToString(),
                                Success = false,
                                Error = ex.Message
                            });
                            result.Success = false;
                            result.ErrorMessage = ex.Message;
                            break;
                        }
                    }
                }

                return result;
            }

            /// <summary>
            /// 并行执行多个算法
            /// </summary>
            private static ExecutionResult ExecuteParallel(
                AlgorithmType[] algorithms,
                List<Tile> tiles,
                int[] experienceMode,
                TerrainAnalysisResult terrainAnalysis)
            {
                var result = new ExecutionResult();
                bool hasSuccess = false;

                foreach (var algorithm in algorithms)
                {
                    try
                    {
                        var metrics = AnalyzeGameplayComplexity(tiles, experienceMode, terrainAnalysis, algorithm);
                        result.Results.Add(new AlgorithmResult
                        {
                            Type = algorithm,
                            Name = algorithm.ToString(),
                            Metrics = metrics,
                            Success = true
                        });
                        hasSuccess = true;
                    }
                    catch (Exception ex)
                    {
                        result.Results.Add(new AlgorithmResult
                        {
                            Type = algorithm,
                            Name = algorithm.ToString(),
                            Success = false,
                            Error = ex.Message
                        });
                    }
                }

                result.Success = hasSuccess;
                if (!hasSuccess)
                {
                    result.ErrorMessage = "所有算法都执行失败";
                }

                return result;
            }

            /// <summary>
            /// 回退执行算法（依序尝试直到成功）
            /// </summary>
            private static ExecutionResult ExecuteFallback(
                AlgorithmType[] algorithms,
                List<Tile> tiles,
                int[] experienceMode,
                TerrainAnalysisResult terrainAnalysis,
                int maxRetryCount)
            {
                var result = new ExecutionResult();

                foreach (var algorithm in algorithms)
                {
                    try
                    {
                        var metrics = AnalyzeGameplayComplexity(tiles, experienceMode, terrainAnalysis, algorithm);
                        result.Results.Add(new AlgorithmResult
                        {
                            Type = algorithm,
                            Name = algorithm.ToString(),
                            Metrics = metrics,
                            Success = true
                        });
                        result.Success = true;
                        return result; // 成功后立即返回
                    }
                    catch (Exception ex)
                    {
                        result.Results.Add(new AlgorithmResult
                        {
                            Type = algorithm,
                            Name = algorithm.ToString(),
                            Success = false,
                            Error = ex.Message
                        });
                    }
                }

                result.Success = false;
                result.ErrorMessage = "所有回退算法都执行失败";
                return result;
            }

            /// <summary>
            /// 执行所有算法并选择最优结果
            /// </summary>
            private static ExecutionResult ExecuteBest(
                AlgorithmType[] algorithms,
                List<Tile> tiles,
                int[] experienceMode,
                TerrainAnalysisResult terrainAnalysis)
            {
                // 复用并行执行逻辑
                return ExecuteParallel(algorithms, tiles, experienceMode, terrainAnalysis);
            }
        }

        /// <summary>
        /// 纯动态分析方法 - 只负责动态指标计算，由BatchLevelEvaluatorSimple调用
        /// </summary>
        /// <param name="tiles">已分配花色的瓦片列表</param>
        /// <param name="experienceMode">体验模式配置</param>
        /// <param name="terrainAnalysis">已计算的地形分析结果</param>
        /// <param name="algorithmType">使用的算法类型</param>
        /// <param name="strategy">可选的自定义策略</param>
        /// <param name="gameEngine">可选的自定义游戏引擎</param>
        /// <param name="metricsCalculator">可选的自定义指标计算器</param>
        /// <returns>动态复杂度指标</returns>
        public static DynamicComplexityMetrics AnalyzeGameplayComplexity(
            List<Tile> tiles,
            int[] experienceMode,
            TerrainAnalysisResult terrainAnalysis,
            AlgorithmType algorithmType = AlgorithmType.OptimalDFS,
            IAutoPlayStrategy strategy = null,
            IGameRulesEngine gameEngine = null,
            IDynamicMetricsCalculator metricsCalculator = null)
        {
            var metrics = new DynamicComplexityMetrics();
            var startTime = DateTime.Now;

            try
            {
                // 使用默认实现或注入的自定义实现
                var engine = gameEngine ?? new AutoPlayBasedGameEngine();
                var autoStrategy = strategy ?? CreateStrategy(algorithmType);
                var calculator = metricsCalculator ?? new DefaultMetricsCalculator();

                // 创建瓦片副本以避免修改原始数据
                var gameplayTiles = CreateGameplayTiles(tiles);

                // 执行动态分析
                return ExecuteDynamicAnalysis(gameplayTiles, engine, autoStrategy, calculator);
            }
            catch (Exception ex)
            {
                Debug.LogError($"动态分析执行失败: {ex.Message}");
                metrics.CompletionStatus = "Error";
                metrics.GameDurationMs = (int)(DateTime.Now - startTime).TotalMilliseconds;
                return metrics;
            }
        }

        /// <summary>
        /// 对比测试方法 - 同时运行两种算法并返回比较结果
        /// </summary>
        /// <param name="tiles">已分配花色的瓦片列表</param>
        /// <param name="experienceMode">体验模式配置</param>
        /// <param name="terrainAnalysis">已计算的地形分析结果</param>
        /// <param name="algorithm1">第一种算法类型</param>
        /// <param name="algorithm2">第二种算法类型</param>
        /// <returns>对比测试结果</returns>
        public static AlgorithmComparisonResult CompareAlgorithms(
            List<Tile> tiles,
            int[] experienceMode,
            TerrainAnalysisResult terrainAnalysis,
            AlgorithmType algorithm1 = AlgorithmType.OptimalDFS,
            AlgorithmType algorithm2 = AlgorithmType.BattleAnalyzer)
        {
            var result = new AlgorithmComparisonResult();
            result.Algorithm1Type = algorithm1;
            result.Algorithm2Type = algorithm2;

            try
            {
                // 运行第一种算法
                var metrics1 = AnalyzeGameplayComplexity(tiles, experienceMode, terrainAnalysis, algorithm1);
                result.Algorithm1Metrics = metrics1;

                // 运行第二种算法
                var metrics2 = AnalyzeGameplayComplexity(tiles, experienceMode, terrainAnalysis, algorithm2);
                result.Algorithm2Metrics = metrics2;

                // 计算对比统计
                result.CalculateComparison();
            }
            catch (Exception ex)
            {
                Debug.LogError($"算法对比测试失败: {ex.Message}");
                result.ErrorMessage = ex.Message;
            }

            return result;
        }

        /// <summary>
        /// 算法对比结果数据结构
        /// </summary>
        [System.Serializable]
        public class AlgorithmComparisonResult
        {
            public AlgorithmType Algorithm1Type { get; set; }
            public AlgorithmType Algorithm2Type { get; set; }
            public DynamicComplexityMetrics Algorithm1Metrics { get; set; }
            public DynamicComplexityMetrics Algorithm2Metrics { get; set; }
            public string ErrorMessage { get; set; } = "";

            // 对比统计
            public int MoveDifference { get; set; } = 0;  // 移动步数差异
            public int TimeDifference { get; set; } = 0;  // 执行时间差异(ms)
            public bool SameResult { get; set; } = false; // 是否得到相同结果
            public string WinnerByMoves { get; set; } = ""; // 移动步数优胜者
            public string WinnerByTime { get; set; } = "";  // 执行时间优胜者

            public void CalculateComparison()
            {
                if (Algorithm1Metrics == null || Algorithm2Metrics == null)
                    return;

                MoveDifference = Algorithm1Metrics.TotalMoves - Algorithm2Metrics.TotalMoves;
                TimeDifference = Algorithm1Metrics.GameDurationMs - Algorithm2Metrics.GameDurationMs;
                SameResult = Algorithm1Metrics.CompletionStatus == Algorithm2Metrics.CompletionStatus &&
                           Algorithm1Metrics.GameCompleted == Algorithm2Metrics.GameCompleted;

                WinnerByMoves = MoveDifference == 0 ? "Tie" :
                              (MoveDifference < 0 ? Algorithm1Type.ToString() : Algorithm2Type.ToString());

                WinnerByTime = TimeDifference == 0 ? "Tie" :
                             (TimeDifference < 0 ? Algorithm1Type.ToString() : Algorithm2Type.ToString());
            }
        }

        /// <summary>
        /// 根据算法类型创建对应的策略实例
        /// </summary>
        private static IAutoPlayStrategy CreateStrategy(AlgorithmType algorithmType)
        {
            return algorithmType switch
            {
                AlgorithmType.OptimalDFS => new OptimalAutoPlayStrategy(),
                AlgorithmType.BattleAnalyzer => new BattleAnalyzerAutoPlayStrategy(),
                AlgorithmType.DefaultGreedy => new DefaultAutoPlayStrategy(),
                _ => new OptimalAutoPlayStrategy()
            };
        }

        #endregion

        #region 私有辅助方法

        /// <summary>
        /// 最优自动游玩策略：在容量阈值T从低到高(0..7)下进行完备搜索。
        /// 首个成功的T即为最小峰值Dock，占用不超过T的最优通关序列。
        /// </summary>
        private class OptimalAutoPlayStrategy : IAutoPlayStrategy
        {
            public string StrategyName => "Optimal-MinPeakDock";

            private Queue<int> plannedSequence;
            private Dictionary<int, TileInfo> tileMap;

            // 公开的求解统计数据（写入动态指标）
            public bool FoundSolution { get; private set; }
            public int MinPeakDock { get; private set; } = -1;
            public long VisitedStates { get; private set; } = 0;
            public long ExpandedNodes { get; private set; } = 0;
            public long SolveTimeMs { get; private set; } = 0;

            private class TileInfo
            {
                public int Id;
                public int Element;
                public int[] Deps;
            }

            public void Initialize(List<Tile> allTiles)
            {
                plannedSequence = new Queue<int>();
                tileMap = new Dictionary<int, TileInfo>(allTiles.Count);
                foreach (var t in allTiles)
                {
                    tileMap[t.ID] = new TileInfo
                    {
                        Id = t.ID,
                        Element = t.ElementValue,
                        Deps = t.Dependencies ?? Array.Empty<int>()
                    };
                }

                var solver = new OptimalSolver(tileMap.Values.ToList(), this);
                var solution = solver.Solve();
                if (solution != null)
                {
                    foreach (var id in solution) plannedSequence.Enqueue(id);
                    FoundSolution = true;
                }
            }

            public Tile SelectNextTile(List<Tile> clickableTiles, GameStateSnapshot currentState)
            {
                if (plannedSequence == null || plannedSequence.Count == 0) return null;
                // 使用预计算序列，确保与引擎一致
                while (plannedSequence.Count > 0)
                {
                    int next = plannedSequence.Peek();
                    var t = clickableTiles.FirstOrDefault(x => x.ID == next);
                    if (t != null)
                    {
                        plannedSequence.Dequeue();
                        return t;
                    }
                    break;
                }
                return null;
            }

            public void Reset()
            {
                plannedSequence?.Clear();
                tileMap?.Clear();
            }

            private class OptimalSolver
            {
                private readonly List<TileInfo> tiles;
                private readonly int n;
                private readonly Dictionary<int, int> idToIndex;
                private readonly Dictionary<int, int> elemToIndex;
                private readonly int[] elemIdx; // per tile index -> element index
                private readonly int[][] depIds; // per tile index -> dep ID array
                private readonly int elemKinds;

                private HashSet<ulong> visited;
                private readonly ulong[] zDesk;
                private readonly ulong[] zDock;
                private readonly System.Random rng = new System.Random(13579);

                private readonly OptimalAutoPlayStrategy parent;
                private long expanded;

                public OptimalSolver(List<TileInfo> allTiles, OptimalAutoPlayStrategy parent)
                {
                    tiles = allTiles;
                    this.parent = parent;
                    n = tiles.Count;
                    // 优化：预分配Dictionary容量避免哈希重建
                    idToIndex = new Dictionary<int, int>(n);
                    var distinctElems = tiles.Select(t => t.Element).Distinct().OrderBy(x => x).ToList();
                    elemKinds = distinctElems.Count;
                    elemToIndex = new Dictionary<int, int>(elemKinds);
                    for (int i = 0; i < elemKinds; i++) elemToIndex[distinctElems[i]] = i;

                    elemIdx = new int[n];
                    depIds = new int[n][];
                    for (int i = 0; i < n; i++)
                    {
                        idToIndex[tiles[i].Id] = i;
                        elemIdx[i] = elemToIndex[tiles[i].Element];
                        depIds[i] = tiles[i].Deps ?? Array.Empty<int>();
                    }

                    // 优化：预分配Zobrist哈希数组
                    zDesk = new ulong[n];
                    for (int i = 0; i < n; i++) zDesk[i] = Next64();
                    zDock = new ulong[elemKinds * 3];
                    for (int i = 0; i < elemKinds * 3; i++) zDock[i] = Next64();
                }

                public List<int> Solve()
                {
                    // 从最低峰值尝试
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int T = 0; T <= 7; T++)
                    {
                        // 优化：根据问题规模预估visited集合大小
                        int estimatedStates = Math.Min(1000000, (int)Math.Pow(2, Math.Min(n, 20)));
                        visited = new HashSet<ulong>(estimatedStates);
                        expanded = 0;
                        var remaining = new bool[n]; 
                        // 优化：直接填充而非循环赋值
                        Array.Fill(remaining, true);
                        var dock = new int[elemKinds];
                        int dockSize = 0;
                        // 优化：预分配确切容量避免扩容
                        var revPath = new List<int>(n);
                        if (Dfs(remaining, dock, ref dockSize, T, revPath))
                        {
                            revPath.Reverse();
                            sw.Stop();
                            parent.MinPeakDock = T;
                            parent.VisitedStates = visited.Count;
                            parent.ExpandedNodes = expanded;
                            parent.SolveTimeMs = sw.ElapsedMilliseconds;
                            return revPath.Select(idx => tiles[idx].Id).ToList();
                        }
                    }
                    sw.Stop();
                    parent.MinPeakDock = -1;
                    parent.VisitedStates = visited?.Count ?? 0;
                    parent.ExpandedNodes = expanded;
                    parent.SolveTimeMs = sw.ElapsedMilliseconds;
                    return null;
                }

                private bool Dfs(bool[] rem, int[] dock, ref int dockSize, int T, List<int> revPath)
                {
                    if (AllCleared(rem))
                    {
                        return dockSize == 0;
                    }

                    var cl = GetClickable(rem);
                    if (cl.Count == 0) return false;
                    
                    // 简单死锁检测：如果Dock已满且无法三消，则死锁
                    if (dockSize >= 7)
                    {
                        bool canTriple = false;
                        for (int e = 0; e < elemKinds; e++)
                        {
                            if (dock[e] % 3 == 2) { canTriple = true; break; }
                        }
                        if (!canTriple) return false;
                    }

                    ulong h = Hash(rem, dock, dockSize);
                    if (visited.Contains(h)) return false;
                    visited.Add(h);
                    expanded++;

                    // 优化：只在有多个候选时才排序，减少不必要的排序开销
                    if (cl.Count > 1)
                    {
                        int currentDockSize = dockSize; // 捕获ref参数值
                        cl.Sort((a, b) => Priority(b, dock, currentDockSize, rem).CompareTo(Priority(a, dock, currentDockSize, rem)));
                    }

                    foreach (var idx in cl)
                    {
                        int e = elemIdx[idx];
                        int before = dock[e];
                        bool triple = (before % 3 == 2);
                        int newDockSize = triple ? (dockSize - 2) : (dockSize + 1);
                        if (newDockSize > T) continue;

                        rem[idx] = false;
                        int oldDockSize = dockSize;
                        dockSize = newDockSize;
                        dock[e] = triple ? (before - 2) : (before + 1);

                        if (Dfs(rem, dock, ref dockSize, T, revPath))
                        {
                            revPath.Add(idx);
                            return true;
                        }

                        // 回溯
                        dock[e] = before;
                        dockSize = oldDockSize;
                        rem[idx] = true;
                    }

                    return false;
                }

                private int Priority(int idx, int[] dock, int dockSize, bool[] rem)
                {
                    int c = dock[elemIdx[idx]] % 3;
                    
                    // 基础优先级
                    if (c == 2) return 30; // 三消：最高优先级，增加权重差距
                    if (c == 1) return 20; // 一对：中等优先级
                    
                    // 新花色：根据Dock压力调整
                    if (dockSize >= 5) return 5;  // Dock接近满时，新花色优先级很低
                    if (dockSize >= 3) return 10; // Dock半满时，新花色优先级降低
                    return 15; // Dock空闲时，新花色正常优先级
                }

                private List<int> GetClickable(bool[] rem)
                {
                    // 优化：预估容量避免动态扩容，但不改变逻辑
                    var res = new List<int>(Math.Min(32, n));
                    for (int i = 0; i < n; i++)
                    {
                        if (!rem[i]) continue;
                        var d = depIds[i];
                        bool ok = true;
                        if (d != null && d.Length > 0)
                        {
                            for (int k = 0; k < d.Length; k++)
                            {
                                if (idToIndex.TryGetValue(d[k], out int j) && rem[j]) { ok = false; break; }
                            }
                        }
                        if (ok) res.Add(i);
                    }
                    return res;
                }

                private bool AllCleared(bool[] rem)
                {
                    // 优化：使用Array.IndexOf可能更快，但保持原逻辑确保100%一致
                    for (int i = 0; i < n; i++) if (rem[i]) return false;
                    return true;
                }

                private ulong Hash(bool[] rem, int[] dock, int dockSize)
                {
                    ulong h = 0UL;
                    // 优化：减少数组访问次数
                    for (int i = 0; i < n; i++) 
                        if (rem[i]) h ^= zDesk[i];
                    
                    // 优化：减少重复计算
                    for (int e = 0; e < elemKinds; e++) 
                        h ^= zDock[e * 3 + (dock[e] % 3)];
                    
                    // 保持原有的哈希混合逻辑完全不变
                    unchecked { h ^= (ulong)(1469598103934665603UL + (ulong)dockSize * 1099511628211UL); }
                    return h;
                }

                private ulong Next64()
                {
                    var b = new byte[8];
                    rng.NextBytes(b);
                    return BitConverter.ToUInt64(b, 0);
                }
            }
        }

        /// <summary>
        /// 创建用于游玩的瓦片副本
        /// </summary>
        private static List<Tile> CreateGameplayTiles(List<Tile> originalTiles)
        {
            var gameplayTiles = new List<Tile>();

            foreach (var originalTile in originalTiles)
            {
                // 创建瓦片副本，避免修改原始数据
                var tileData = new TileData
                {
                    ID = originalTile.ID,
                    Layer = originalTile.LayerID,
                    Dependencies = originalTile.Dependencies,
                    IsConst = originalTile.TileConfig.IsConst,
                    ConstElementValue = originalTile.TileConfig.ConstElementValue,
                    PosX = originalTile.TileConfig.PosX,
                    PosY = originalTile.TileConfig.PosY
                };

                var gameTile = new Tile(tileData);
                gameTile.SetElementValue(originalTile.ElementValue);
                gameplayTiles.Add(gameTile);
            }

            return gameplayTiles;
        }

        /// <summary>
        /// 执行动态分析核心逻辑 - 自动游玩并收集指标
        /// </summary>
        private static DynamicComplexityMetrics ExecuteDynamicAnalysis(
            List<Tile> tiles,
            IGameRulesEngine gameEngine,
            IAutoPlayStrategy strategy,
            IDynamicMetricsCalculator metricsCalculator)
        {
            var metrics = new DynamicComplexityMetrics();
            var startTime = DateTime.Now;

            try
            {
                // 初始化游戏组件
                gameEngine.InitializeGame(tiles);
                strategy.Initialize(tiles);
                metricsCalculator.Reset();

                // 执行自动游玩循环
                GameStateSnapshot previousState = null;
                const int maxMoves = 1000; // 防止无限循环
                int moveCount = 0;

                while (!gameEngine.IsGameOver() && moveCount < maxMoves)
                {
                    var currentState = gameEngine.GetCurrentState();

                    // 收集状态变化数据
                    if (previousState != null)
                    {
                        metricsCalculator.ProcessStateChange(previousState, currentState);
                    }

                    // 获取可点击瓦片
                    var clickableTiles = gameEngine.GetClickableTiles();
                    if (clickableTiles.Count == 0) break;

                    // 策略选择下一步
                    var selectedTile = strategy.SelectNextTile(clickableTiles, currentState);
                    if (selectedTile == null) break;

                    // 执行移动
                    if (!gameEngine.ExecuteTileClick(selectedTile)) break;

                    // 记录该步的tileId与执行后Dock数量
                    try
                    {
                        metrics.OptimalMoveTileIds.Add(selectedTile.ID);
                        var afterState = gameEngine.GetCurrentState();
                        int dockCount = (afterState != null && afterState.DockTiles != null) ? afterState.DockTiles.Count : 0;
                        metrics.DockCountPerMove.Add(dockCount);
                        if (dockCount > metrics.PeakDockDuringSolution) metrics.PeakDockDuringSolution = dockCount;
                    }
                    catch { }

                    previousState = currentState;
                    moveCount++;
                }

                // 计算最终指标
                metrics.GameCompleted = gameEngine.IsGameWon();
                metrics.CompletionStatus = gameEngine.IsGameWon() ? "Victory" :
                                         gameEngine.IsGameOver() ? "Defeat" : "Timeout";
                metrics.GameDurationMs = (int)(DateTime.Now - startTime).TotalMilliseconds;

                // 让指标计算器计算具体的复杂度指标
                metricsCalculator.CalculateFinalMetrics(metrics);

                // 写入最优求解器统计数据（若使用了最优策略）
                if (strategy is OptimalAutoPlayStrategy opt)
                {
                    if (opt.FoundSolution)
                    {
                        metrics.AddMetric("MinPeakDock", opt.MinPeakDock);
                    }
                    else
                    {
                        metrics.AddMetric("MinPeakDock", -1);
                    }
                    metrics.AddMetric("ExpandedNodes", (int)Math.Min(int.MaxValue, opt.ExpandedNodes));
                    metrics.AddMetric("SolveTimeMs", (int)Math.Min(int.MaxValue, opt.SolveTimeMs));
                }
                // 写入BattleAnalyzer统计数据（若使用了BattleAnalyzer策略）
                else if (strategy is BattleAnalyzerAutoPlayStrategy battleAnalyzer)
                {
                    metrics.AddMetric("AnalysisTimeMs", (int)Math.Min(int.MaxValue, battleAnalyzer.AnalysisTimeMs));
                    metrics.AddMetric("TotalAnalysisCalls", battleAnalyzer.TotalAnalysisCalls);
                    metrics.AddMetric("SuccessfulMoves", battleAnalyzer.SuccessfulMoves);
                    metrics.AddMetric("AnalysisSuccessRate",
                        battleAnalyzer.TotalAnalysisCalls > 0 ?
                        (double)battleAnalyzer.SuccessfulMoves / battleAnalyzer.TotalAnalysisCalls : 0.0);
                }

                return metrics;
            }
            catch (Exception ex)
            {
                metrics.CompletionStatus = "Error";
                metrics.GameDurationMs = (int)(DateTime.Now - startTime).TotalMilliseconds;
                Debug.LogError($"动态分析执行失败: {ex.Message}");
                return metrics;
            }
        }
        #endregion
    }
}
