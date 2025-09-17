using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using DGuo.Client.TileMatch;
using DGuo.Client.TileMatch.DesignerAlgo.RuleBasedAlgo;
using DGuo.Client.TileMatch.DesignerAlgo.Evaluation;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DGuo.Client.TileMatch.EditorTools
{
    public static class TM_BatchBattleAnalyzer
    {
        private const int MaxMoves = 500;
        private const int FixedSeed = 12345678;

        private class CsvRow
        {
            public int TerrainId;
            public int[] ExpFix1;
            public int TypeCount1;
        }

        private class RunMetrics
        {
            public int TerrainId;
            public string LevelName;
            public string Exp;
            public int ColorCount;
            public int TotalMoves;
            public string Result;
            public int PeakDock;
            public long AnalysisTimeMs;
            public int AnalysisCalls;
            public int SuccessfulMoves;
            public string MoveSeq;
            public string DockSeq;
            public int DurationMs;
            public string Error;
        }

        [UnityEditor.MenuItem("TileMatch/分析/批量运行 BattleAnalyzer (CSV)")]
        public static void RunFromCsv()
        {
            try
            {
                var csvPath = Path.Combine(Application.dataPath, "_Editor/all_level.csv");
                var rows = LoadCsv(csvPath);
                if (rows.Count == 0)
                {
                    UnityEngine.Debug.LogError($"CSV 为空或未找到: {csvPath}");
                    return;
                }

                var outDir = Path.Combine(Application.dataPath, "_Editor/Outputs");
                Directory.CreateDirectory(outDir);
                var ts = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var outCsv = Path.Combine(outDir, $"BattleAnalyzer_{ts}.csv");

                var results = new List<RunMetrics>();

                int idx = 0;
                foreach (var row in rows)
                {
                    idx++;
                    var levelName = $"Level_{row.TerrainId:D3}";
                    UnityEngine.Debug.Log($"[{idx}/{rows.Count}] 运行 {levelName} exp={Fmt(row.ExpFix1)} colors={row.TypeCount1}");
                    results.Add(RunOne(levelName, row.TerrainId, row.ExpFix1, row.TypeCount1));
                }

                ExportCsv(results, outCsv);
                UnityEngine.Debug.Log($"导出完成: {outCsv}");
                try { System.Diagnostics.Process.Start("explorer.exe", outDir.Replace('/', '\\')); } catch { }
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogError($"批量运行失败: {ex.Message}\n{ex.StackTrace}");
            }
        }

                // ===== 离线 Battle 视图 + 分析器（完全照搬 TileMatchBattleAnalyzerMgr 逻辑） =====
        private class OfflineBattle
        {
            public Dictionary<int, Tile> AllTiles = new Dictionary<int, Tile>();
        }

        private class AnalyzerMirror // 复刻 TileMatchBattleAnalyzerMgr 关键逻辑
        {
            private OfflineBattle _battle;
            private Dictionary<int, List<MatchGroup>> _matchGroups;
            public AnalyzerMirror(OfflineBattle b) { _battle = b; }
            public void SetElementValues(List<int> values)
            {
                _matchGroups = new Dictionary<int, List<MatchGroup>>();
                foreach (var v in values) _matchGroups[v] = null;
            }
            public void Analyze()
            {
                if (_matchGroups == null) return;
                var keys = _matchGroups.Keys.ToList();
                foreach (var k in keys) _matchGroups[k] = new List<MatchGroup>();
                foreach (var ev in keys) _getAllMatchs(ev);
            }
            private void _getAllMatchs(int elementValue)
            {
                var allElementValueTiles = _battle.AllTiles.Values
                    .Where(t => t.ElementValue == elementValue && !t.IsSetFlag(ETileFlag.Destroyed))
                    .ToList();
                if (allElementValueTiles.Count < 3) return;
                allElementValueTiles.Sort((a,b) => {
                    int da = a.PileType == PileType.Dock ? 0 : (a.runtimeDependencies?.Count ?? 0) + 1;
                    int db = b.PileType == PileType.Dock ? 0 : (b.runtimeDependencies?.Count ?? 0) + 1;
                    return da.CompareTo(db);
                });
                var groups = new List<MatchGroup>();
                for (int i=0; i<allElementValueTiles.Count-2; i++)
                for (int j=i+1; j<allElementValueTiles.Count-1; j++)
                for (int k=j+1; k<allElementValueTiles.Count; k++)
                {
                    var matchTiles = new List<Tile>{ allElementValueTiles[i], allElementValueTiles[j], allElementValueTiles[k] };
                    int cost = _calculateCost(matchTiles, out var path);
                    groups.Add(new MatchGroup{ matchTiles = matchTiles, totalCost = cost, path = path });
                }
                groups.Sort((x,y) => x.totalCost.CompareTo(y.totalCost));
                _matchGroups[elementValue] = groups;
            }
            private int _calculateCost(List<Tile> matchTiles, out HashSet<int> path)
            {
                var allDeps = new HashSet<int>();
                foreach (var tile in matchTiles)
                {
                    if (tile.PileType == PileType.Dock) continue;
                    _collectAllDependencies(tile, allDeps);
                    allDeps.Add(tile.ID);
                }
                path = allDeps; return allDeps.Count;
            }
            private void _collectAllDependencies(Tile tile, HashSet<int> set)
            {
                if (tile == null || tile.runtimeDependencies == null || tile.runtimeDependencies.Count == 0) return;
                foreach (var depId in tile.runtimeDependencies)
                {
                    if (set.Add(depId) && _battle.AllTiles.TryGetValue(depId, out var depTile))
                    {
                        _collectAllDependencies(depTile, set);
                    }
                }
            }
            public List<MatchGroup> GetMatchGroups(int ev)
            {
                if (_matchGroups == null || !_matchGroups.ContainsKey(ev)) return new List<MatchGroup>();
                return _matchGroups[ev] ?? new List<MatchGroup>();
            }
            public class MatchGroup
            {
                public List<Tile> matchTiles;
                public int totalCost;
                public HashSet<int> path;
            }
        }

        // 简化引擎：Desk/Dock/Destroyed 状态 + 运行时依赖维护，其他规则交给 Rule
        private class OfflineEngine
        {
            public List<Tile> Desk = new List<Tile>();
            public List<Tile> Dock = new List<Tile>();
            public int MoveCount { get; private set; }
            public int PeakDock { get; private set; }

            public void Init(List<Tile> tiles)
            {
                Desk = tiles.Where(t => !t.IsSetFlag(ETileFlag.Destroyed)).ToList();
                Dock.Clear(); MoveCount = 0; PeakDock = 0;
                foreach (var t in Desk) { t.PileType = PileType.Desk; }
                RecomputeRuntimeDeps();
            }
            public void RecomputeRuntimeDeps()
            {
                var deskIds = Desk.Select(t => t.ID).ToHashSet();
                foreach (var t in Desk.Concat(Dock))
                {
                    if (t.IsSetFlag(ETileFlag.Destroyed)) { t.runtimeDependencies = new List<int>(); continue; }
                    var deps = t.Dependencies ?? Array.Empty<int>();
                    t.runtimeDependencies = deps.Where(id => deskIds.Contains(id)).ToList();
                }
            }
            public List<Tile> GetClickable()
            {
                return Desk.Where(t => !t.IsSetFlag(ETileFlag.Destroyed) && (t.runtimeDependencies?.Count ?? 0) == 0).ToList();
            }
            public bool Click(Tile tile)
            {
                if (tile == null) return false;
                if (!(tile.runtimeDependencies == null || tile.runtimeDependencies.Count == 0)) return false;
                // 移入 Dock
                Desk.Remove(tile);
                tile.PileType = PileType.Dock;
                Dock.Add(tile);
                MoveCount++;
                // 排序行为对三消次序有影响：按 Dock.SortDockTiles 逻辑实现
                if (Dock.Count > 1)
                {
                    var groups = new Dictionary<int, List<Tile>>();
                    var order = new List<int>();
                    foreach (var t in Dock)
                    {
                        if (!groups.TryGetValue(t.ElementValue, out var list)) { list = new List<Tile>(); groups[t.ElementValue] = list; order.Add(t.ElementValue); }
                        list.Add(t);
                    }
                    var sorted = new List<Tile>(Dock.Count);
                    foreach (var key in order) sorted.AddRange(groups[key]);
                    Dock = sorted;
                }
                // 按 Rule 检查并仅消除一个三消
                if (Rule.CheckDockMatch(Dock, out var matched) && matched != null)
                {
                    foreach (var t in matched)
                    {
                        Dock.Remove(t);
                        t.SetFlag(ETileFlag.Destroyed);
                    }
                }
                PeakDock = Math.Max(PeakDock, Dock.Count);
                // 依赖重算
                RecomputeRuntimeDeps();
                return true;
            }
            public bool IsWin() => Desk.Count == 0; // 所有 Desk tile 都被拿走（消除或在 Dock）
            public bool IsLose() => Dock.Count >= 7 || (GetClickable().Count == 0 && !IsWin());
        }        private static RunMetrics RunOne(string levelName, int terrainId, int[] exp, int colorCount)
        {
            var swAll = Stopwatch.StartNew();
            var metrics = new RunMetrics
            {
                TerrainId = terrainId,
                LevelName = ConvertToJsonLevelName(levelName),
                Exp = $"[{exp[0]},{exp[1]},{exp[2]}]",
                ColorCount = colorCount
            };

            try
            {
                var levelData = LoadLevelData(levelName);
                if (levelData == null) { metrics.Result = "Error"; metrics.Error = "LevelData null"; return metrics; }

                var tiles = CreateTileListFromLevelData(levelData);

                UnityEngine.Random.InitState(FixedSeed);
                var algo = new RuleBasedAlgorithm();
                algo.InitializeRandomSeed(FixedSeed);
                var availableColors = CreateAvailableColors(Mathf.Max(1, colorCount));
                algo.AssignTileTypes(tiles, exp, availableColors);

                // 使用离线引擎 + AnalyzerMirror（完全照搬 TileMatchBattleAnalyzerMgr 的组合与代价逻辑）
                var offlineBattle = new OfflineBattle();
                foreach (var t in tiles) offlineBattle.AllTiles[t.ID] = t;
                var engine = new OfflineEngine();
                engine.Init(tiles);

                var moveSeq = new List<int>();
                var dockSeq = new List<int>();

                int safeGuard = 2000; // 防护上限
                while (!engine.IsWin() && !engine.IsLose() && engine.MoveCount < MaxMoves && safeGuard-- > 0)
                {
                    engine.RecomputeRuntimeDeps();
                    var clickable = engine.GetClickable();
                    if (clickable.Count == 0) break;

                    var analyzer = new AnalyzerMirror(offlineBattle);
                    var elementValues = tiles.Where(t => !t.IsSetFlag(ETileFlag.Destroyed)).Select(t => t.ElementValue).Distinct().ToList();
                    analyzer.SetElementValues(elementValues);
                    analyzer.Analyze();

                    Tile best = null; int bestCost = int.MaxValue;
                    foreach (var c in clickable)
                    {
                        var groups = analyzer.GetMatchGroups(c.ElementValue);
                        var g = groups.FirstOrDefault(x => x.matchTiles.Any(mt => mt.ID == c.ID));
                        if (g != null && g.totalCost < bestCost)
                        {
                            bestCost = g.totalCost; best = c;
                        }
                    }
                    if (best == null) best = clickable.FirstOrDefault();
                    if (!engine.Click(best)) break;

                    moveSeq.Add(best.ID);
                    dockSeq.Add(engine.Dock.Count);
                }

                metrics.TotalMoves = engine.MoveCount;
                metrics.Result = engine.IsWin() ? "Victory" : "Defeat";
                metrics.PeakDock = engine.PeakDock;
                metrics.AnalysisTimeMs = 0;
                metrics.AnalysisCalls = 0;
                metrics.SuccessfulMoves = 0;
                metrics.MoveSeq = string.Join(" ", moveSeq);
                metrics.DockSeq = string.Join(",", dockSeq);
            }
            catch (Exception ex)
            {
                metrics.Result = "Error";
                metrics.Error = ex.Message;
            }
            finally
            {
                swAll.Stop();
                metrics.DurationMs = (int)swAll.ElapsedMilliseconds;
            }
            return metrics;
        }

        private static List<CsvRow> LoadCsv(string path)
        {
            var list = new List<CsvRow>();
            if (!File.Exists(path)) return list;
            var lines = File.ReadAllLines(path);
            for (int i = 1; i < lines.Length; i++)
            {
                var parts = ParseCsvLine(lines[i]);
                if (parts.Length < 7) continue;
                if (!int.TryParse(parts[0].Trim(), out int id)) continue;
                var row = new CsvRow
                {
                    TerrainId = id,
                    ExpFix1 = ParseIntArray(parts[1]),
                    TypeCount1 = ParseIntOrDefault(parts[4], 7)
                };
                if (row.ExpFix1 == null || row.ExpFix1.Length < 3) row.ExpFix1 = new[] {1,1,1};
                list.Add(row);
            }
            return list;
        }

        private static string[] ParseCsvLine(string line)
        {
            var res = new List<string>();
            var sb = new StringBuilder(); bool inQ = false;
            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"') { inQ = !inQ; continue; }
                if (c == ',' && !inQ) { res.Add(sb.ToString()); sb.Length = 0; continue; }
                sb.Append(c);
            }
            res.Add(sb.ToString());
            return res.ToArray();
        }

        private static int[] ParseIntArray(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return new[] {1,1,1};
            s = s.Trim().Trim('[',']','{','}','(',')');
            if (string.IsNullOrWhiteSpace(s)) return new[] {1,1,1};
            if (int.TryParse(s, out int single)) return new[] {single, single, single};
            var parts = s.Split(new[] {',',';','\t',' '}, StringSplitOptions.RemoveEmptyEntries);
            var vals = new List<int>();
            foreach (var p in parts) if (int.TryParse(p.Trim(), out int v)) vals.Add(v);
            if (vals.Count == 0) return new[] {1,1,1};
            if (vals.Count == 1) return new[] {vals[0], vals[0], vals[0]};
            if (vals.Count == 2) return new[] {vals[0], vals[1], vals[1]};
            return vals.Take(3).ToArray();
        }

        private static int ParseIntOrDefault(string s, int d)
        {
            return int.TryParse((s ?? string.Empty).Trim(), out int v) ? v : d;
        }

        private static LevelData LoadLevelData(string levelName)
        {
            try
            {
                string jsonFileName = levelName.StartsWith("Level_")
                    ? ($"{100000 + int.Parse(levelName.Substring(6))}.json")
                    : (levelName.EndsWith(".json") ? levelName : levelName + ".json");
                string jsonPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Tools", "Config", "Json", "Levels", jsonFileName));
                if (!File.Exists(jsonPath)) { UnityEngine.Debug.LogError($"未找到关卡 JSON: {jsonPath}"); return null; }
                var json = File.ReadAllText(jsonPath);
                return JsonUtility.FromJson<LevelData>(json);
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogError($"加载关卡失败 {levelName}: {ex.Message}");
                return null;
            }
        }

        private static List<Tile> CreateTileListFromLevelData(LevelData levelData)
        {
            var tiles = new List<Tile>();
            foreach (var layer in levelData.Layers)
            {
                foreach (var td in layer.tiles)
                {
                    var t = new Tile(td);
                    if (td.IsConst) t.SetElementValue(td.ConstElementValue);
                    tiles.Add(t);
                }
            }
            return tiles;
        }

        private static List<int> CreateAvailableColors(int count)
        {
            var all = GetAvailableElementValues();
            if (count <= all.Length) return all.Take(count).ToList();
            return all.ToList();
        }

        private static int[] cachedColors;
        private static int[] GetAvailableElementValues()
        {
            if (cachedColors != null) return cachedColors;
            try
            {
                var db = UnityEngine.Resources.Load<LevelDatabase>("StaticSettings/LevelDatabase");
                if (db != null && db.Tiles != null)
                {
                    cachedColors = db.Tiles.Where(x => x != null && x.ElementValue > 0)
                                           .Select(x => x.ElementValue)
                                           .Distinct().OrderBy(x => x).ToArray();
                    return cachedColors;
                }
            }
            catch { }
            cachedColors = new int[] {101,102,103,201,202,301,302,401,402,403,501,502,601,602,701,702,703,801,802};
            return cachedColors;
        }

        private static string ConvertToJsonLevelName(string levelName)
        {
            if (levelName.StartsWith("Level_") && int.TryParse(levelName.Substring(6), out int n))
                return (100000 + n).ToString();
            return levelName;
        }

        private static string Fmt(int[] a) => a == null ? "[]" : $"[{string.Join(",", a)}]";

        private static void ExportCsv(List<RunMetrics> rows, string path)
        {
            var sb = new StringBuilder();
            sb.AppendLine("TerrainId,LevelName,Exp,ColorCount,TotalMoves,Result,PeakDock,AnalysisTimeMs,AnalysisCalls,SuccessfulMoves,DurationMs,MoveSeq,DockSeq,Error");
            foreach (var r in rows)
            {
                sb.AppendLine(string.Join(",",
                    r.TerrainId.ToString(),
                    Escape(r.LevelName),
                    Escape(r.Exp),
                    r.ColorCount.ToString(),
                    r.TotalMoves.ToString(),
                    r.Result ?? "",
                    r.PeakDock.ToString(),
                    r.AnalysisTimeMs.ToString(),
                    r.AnalysisCalls.ToString(),
                    r.SuccessfulMoves.ToString(),
                    r.DurationMs.ToString(),
                    Escape(r.MoveSeq ?? ""),
                    Escape(r.DockSeq ?? ""),
                    Escape(r.Error ?? "")
                ));
            }
            File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
        }

        private static string Escape(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.IndexOfAny(new[] {',','\"','\n','\r'}) >= 0)
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }
    }
}






