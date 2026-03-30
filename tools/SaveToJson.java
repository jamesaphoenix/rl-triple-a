package tools;

import games.strategy.engine.data.*;
import games.strategy.engine.framework.GameDataManager;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Extract game state from a TripleA .tsvg save file as JSON.
 *
 * Usage: java -cp <classpath> tools.SaveToJson <input.tsvg> [output.json]
 */
public class SaveToJson {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: SaveToJson <save.tsvg> [output.json]");
            System.exit(1);
        }

        Path savePath = Path.of(args[0]);
        System.err.println("Loading: " + savePath);

        Optional<GameData> optData = GameDataManager.loadGame(savePath);
        if (optData.isEmpty()) {
            System.err.println("Failed to load save file!");
            System.exit(1);
        }

        GameData data = optData.get();
        String json = toJson(data);

        if (args.length > 1) {
            Files.writeString(Path.of(args[1]), json);
            System.err.println("Written to: " + args[1]);
        } else {
            System.out.println(json);
        }
    }

    static String toJson(GameData data) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");

        // Game info
        sb.append("  \"gameName\": ").append(quote(data.getGameName())).append(",\n");

        // Round
        sb.append("  \"round\": ").append(data.getSequence().getRound()).append(",\n");

        // Players
        sb.append("  \"players\": {\n");
        boolean firstP = true;
        for (GamePlayer player : data.getPlayerList().getPlayers()) {
            if (!firstP) sb.append(",\n");
            firstP = false;
            int pus = player.getResources().getQuantity("PUs");
            sb.append("    ").append(quote(player.getName())).append(": {\"pus\": ").append(pus).append("}");
        }
        sb.append("\n  },\n");

        // Territories
        sb.append("  \"territories\": {\n");
        boolean firstT = true;
        for (Territory t : data.getMap().getTerritories()) {
            if (!firstT) sb.append(",\n");
            firstT = false;

            sb.append("    ").append(quote(t.getName())).append(": {\n");
            sb.append("      \"owner\": ");
            if (t.getOwner() != null && !t.getOwner().isNull()) {
                sb.append(quote(t.getOwner().getName()));
            } else {
                sb.append("null");
            }
            sb.append(",\n");
            sb.append("      \"isWater\": ").append(t.isWater()).append(",\n");

            // Units grouped by owner and type
            sb.append("      \"units\": {");
            Map<String, Map<String, Integer>> unitsByPlayer = new LinkedHashMap<>();
            for (Unit unit : t.getUnitCollection()) {
                String ownerName = unit.getOwner().getName();
                String typeName = unit.getType().getName();
                unitsByPlayer
                    .computeIfAbsent(ownerName, k -> new LinkedHashMap<>())
                    .merge(typeName, 1, Integer::sum);
            }

            boolean firstUP = true;
            for (var entry : unitsByPlayer.entrySet()) {
                if (!firstUP) sb.append(",");
                firstUP = false;
                sb.append("\n        ").append(quote(entry.getKey())).append(": {");
                boolean firstU = true;
                for (var ue : entry.getValue().entrySet()) {
                    if (!firstU) sb.append(", ");
                    firstU = false;
                    sb.append(quote(ue.getKey())).append(": ").append(ue.getValue());
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

    static String quote(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
