/**
 * Extract game state from TripleA .tsvg save files as JSON.
 *
 * This uses TripleA's own GameDataManager to properly deserialize
 * the save files, then exports territory ownership, unit positions,
 * and player resources as JSON for Python consumption.
 *
 * Build: javac -cp <triplea-classpath> SaveGameExtractor.java
 * Run:   java -cp <triplea-classpath>:. SaveGameExtractor <save-file.tsvg> <output.json>
 *
 * NOTE: This requires TripleA's game-core JAR on the classpath.
 * Build TripleA first: cd triplea && ./gradlew :game-app:game-core:jar
 */

import games.strategy.engine.data.*;
import games.strategy.engine.framework.GameDataManager;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class SaveGameExtractor {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: SaveGameExtractor <save-file.tsvg> [output.json]");
            System.err.println("  If output.json is omitted, prints to stdout");
            System.exit(1);
        }

        Path savePath = Path.of(args[0]);
        String outputPath = args.length > 1 ? args[1] : null;

        System.err.println("Loading save: " + savePath);

        Optional<GameData> optData = GameDataManager.loadGame(savePath);
        if (optData.isEmpty()) {
            System.err.println("Failed to load save file!");
            System.exit(1);
        }

        GameData data = optData.get();
        String json = exportToJson(data);

        if (outputPath != null) {
            Files.writeString(Path.of(outputPath), json);
            System.err.println("Written to: " + outputPath);
        } else {
            System.out.println(json);
        }
    }

    static String exportToJson(GameData data) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");

        // Game info
        sb.append("  \"gameName\": \"").append(escape(data.getGameName())).append("\",\n");

        // Current round and step
        GameSequence seq = data.getSequence();
        sb.append("  \"round\": ").append(seq.getRound()).append(",\n");

        // Players and resources
        sb.append("  \"players\": {\n");
        boolean firstPlayer = true;
        for (GamePlayer player : data.getPlayerList().getPlayers()) {
            if (!firstPlayer) sb.append(",\n");
            firstPlayer = false;
            sb.append("    \"").append(escape(player.getName())).append("\": {\n");
            sb.append("      \"pus\": ").append(player.getResources().getQuantity("PUs")).append("\n");
            sb.append("    }");
        }
        sb.append("\n  },\n");

        // Territories
        sb.append("  \"territories\": {\n");
        boolean firstTerritory = true;
        for (Territory t : data.getMap().getTerritories()) {
            if (!firstTerritory) sb.append(",\n");
            firstTerritory = false;
            sb.append("    \"").append(escape(t.getName())).append("\": {\n");
            sb.append("      \"owner\": ").append(t.getOwner() != null ?
                "\"" + escape(t.getOwner().getName()) + "\"" : "null").append(",\n");
            sb.append("      \"isWater\": ").append(t.isWater()).append(",\n");

            // Units in this territory
            sb.append("      \"units\": {");
            Map<String, Map<String, Integer>> unitsByPlayer = new LinkedHashMap<>();
            for (Unit unit : t.getUnitCollection()) {
                String ownerName = unit.getOwner().getName();
                String typeName = unit.getType().getName();
                unitsByPlayer
                    .computeIfAbsent(ownerName, k -> new LinkedHashMap<>())
                    .merge(typeName, 1, Integer::sum);
            }

            boolean firstUnitPlayer = true;
            for (Map.Entry<String, Map<String, Integer>> entry : unitsByPlayer.entrySet()) {
                if (!firstUnitPlayer) sb.append(",");
                firstUnitPlayer = false;
                sb.append("\n        \"").append(escape(entry.getKey())).append("\": {");
                boolean firstUnit = true;
                for (Map.Entry<String, Integer> ue : entry.getValue().entrySet()) {
                    if (!firstUnit) sb.append(", ");
                    firstUnit = false;
                    sb.append("\"").append(escape(ue.getKey())).append("\": ").append(ue.getValue());
                }
                sb.append("}");
            }
            if (!unitsByPlayer.isEmpty()) sb.append("\n      ");
            sb.append("}\n");
            sb.append("    }");
        }
        sb.append("\n  }\n");

        sb.append("}\n");
        return sb.toString();
    }

    static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
