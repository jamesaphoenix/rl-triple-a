package games.strategy.triplea.ai.rl;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GameMap;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.GameState;
import games.strategy.engine.data.MoveDescription;
import games.strategy.engine.data.NamedAttachable;
import games.strategy.engine.data.ProductionFrontier;
import games.strategy.engine.data.ProductionRule;
import games.strategy.engine.data.Resource;
import games.strategy.engine.data.Route;
import games.strategy.engine.data.Territory;
import games.strategy.engine.data.Unit;
import games.strategy.triplea.Constants;
import games.strategy.triplea.ai.AbstractAi;
import games.strategy.triplea.attachments.TerritoryAttachment;
import games.strategy.triplea.delegate.remote.IAbstractPlaceDelegate;
import games.strategy.triplea.delegate.remote.IAbstractPlaceDelegate.BidMode;
import games.strategy.triplea.delegate.remote.IMoveDelegate;
import games.strategy.triplea.delegate.remote.IPurchaseDelegate;
import games.strategy.triplea.delegate.remote.ITechDelegate;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.collections.IntegerMap;

/**
 * An RL (Reinforcement Learning) bot for TripleA.
 *
 * <p>On each game phase (purchase, move, place, tech), this bot serializes the current game state
 * to JSON and sends an HTTP POST to a local Python action server at localhost:8080/api/action. The
 * server returns action decisions as JSON, which this bot translates into delegate calls.
 *
 * <p>If the HTTP server is unreachable or returns an error, the bot gracefully falls back to doing
 * nothing for that phase.
 */
@Slf4j
public class RLBot extends AbstractAi {

  private static final String ACTION_SERVER_URL = "http://localhost:8080/api/action";
  private static final int HTTP_TIMEOUT_MS = 10_000;

  public RLBot(final String name, final String playerLabel) {
    super(name, playerLabel);
  }

  // ---------------------------------------------------------------------------
  // Phase implementations
  // ---------------------------------------------------------------------------

  @Override
  protected void purchase(
      final boolean purchaseForBid,
      final int pusToSpend,
      final IPurchaseDelegate purchaseDelegate,
      final GameData data,
      final GamePlayer player) {

    if (pusToSpend <= 0) {
      return;
    }

    try {
      final String gameStateJson = serializeGameState(data, player, "purchase");
      final String responseJson = postToActionServer(gameStateJson);
      if (responseJson == null) {
        log.warn("RLBot: no response from action server during purchase, skipping.");
        return;
      }
      executePurchaseActions(responseJson, purchaseDelegate, data, player, purchaseForBid);
    } catch (final Exception e) {
      log.warn("RLBot: error during purchase phase, falling back to no-op.", e);
    }
  }

  @Override
  protected void move(
      final boolean nonCombat,
      final IMoveDelegate moveDel,
      final GameData data,
      final GamePlayer player) {

    try {
      final String phase = nonCombat ? "nonCombatMove" : "combatMove";
      final String gameStateJson = serializeGameState(data, player, phase);
      final String responseJson = postToActionServer(gameStateJson);
      if (responseJson == null) {
        log.warn("RLBot: no response from action server during move, skipping.");
        return;
      }
      executeMoveActions(responseJson, moveDel, data, player);
    } catch (final Exception e) {
      log.warn("RLBot: error during move phase, falling back to no-op.", e);
    }
  }

  @Override
  protected void place(
      final boolean placeForBid,
      final IAbstractPlaceDelegate placeDelegate,
      final GameState data,
      final GamePlayer player) {

    try {
      final String gameStateJson = serializeGameState((GameData) data, player, "place");
      final String responseJson = postToActionServer(gameStateJson);
      if (responseJson == null) {
        log.warn("RLBot: no response from action server during place, skipping.");
        return;
      }
      executePlaceActions(responseJson, placeDelegate, (GameData) data, player, placeForBid);
    } catch (final Exception e) {
      log.warn("RLBot: error during place phase, falling back to no-op.", e);
    }
  }

  @Override
  protected void tech(
      final ITechDelegate techDelegate, final GameData data, final GamePlayer player) {
    // Tech is typically skipped by RL bots -- no investment in tech rolls.
    // We still call the server in case the model wants to roll for tech.
    try {
      final String gameStateJson = serializeGameState(data, player, "tech");
      final String responseJson = postToActionServer(gameStateJson);
      if (responseJson == null) {
        log.info("RLBot: no response from action server during tech, skipping.");
        return;
      }
      // The server can return {"skip": true} to skip tech, which is the default.
      // Otherwise we ignore tech actions for now.
      log.info("RLBot: tech phase response received (no-op for now).");
    } catch (final Exception e) {
      log.warn("RLBot: error during tech phase, falling back to no-op.", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Game state serialization
  // ---------------------------------------------------------------------------

  /**
   * Serializes the current game state into a JSON string that the Python action server can
   * interpret. We build JSON manually to avoid adding a Gson/Jackson dependency.
   */
  private String serializeGameState(
      final GameData data, final GamePlayer player, final String phase) {

    final StringBuilder sb = new StringBuilder(8192);
    sb.append('{');

    // Phase info
    appendJsonString(sb, "phase", phase);
    sb.append(',');
    appendJsonString(sb, "currentPlayer", player.getName());
    sb.append(',');
    sb.append("\"round\":").append(data.getSequence().getRound());
    sb.append(',');

    // Player resources (PUs)
    final Resource pus = data.getResourceList().getResourceOrThrow(Constants.PUS);
    sb.append("\"pus\":").append(player.getResources().getQuantity(pus));
    sb.append(',');

    // Territories
    sb.append("\"territories\":[");
    final List<Territory> territories = data.getMap().getTerritories();
    boolean firstTerritory = true;
    for (final Territory t : territories) {
      if (!firstTerritory) {
        sb.append(',');
      }
      firstTerritory = false;
      sb.append('{');
      appendJsonString(sb, "name", t.getName());
      sb.append(',');
      appendJsonString(sb, "owner", t.getOwner().getName());
      sb.append(',');
      sb.append("\"isWater\":").append(t.isWater());
      sb.append(',');

      // Production value
      final int production = TerritoryAttachment.getProduction(t);
      sb.append("\"production\":").append(production);
      sb.append(',');

      // Units on this territory
      sb.append("\"units\":[");
      boolean firstUnit = true;
      for (final Unit u : t.getUnitCollection()) {
        if (!firstUnit) {
          sb.append(',');
        }
        firstUnit = false;
        sb.append('{');
        appendJsonString(sb, "type", u.getType().getName());
        sb.append(',');
        appendJsonString(sb, "owner", u.getOwner().getName());
        sb.append(',');
        sb.append("\"hits\":").append(u.getHits());
        sb.append('}');
      }
      sb.append(']');

      // Neighboring territories
      sb.append(",\"neighbors\":[");
      boolean firstNeighbor = true;
      for (final Territory neighbor : data.getMap().getNeighbors(t)) {
        if (!firstNeighbor) {
          sb.append(',');
        }
        firstNeighbor = false;
        sb.append('"').append(escapeJson(neighbor.getName())).append('"');
      }
      sb.append(']');

      sb.append('}');
    }
    sb.append(']');
    sb.append(',');

    // Available production rules for purchase phase
    sb.append("\"productionRules\":[");
    final ProductionFrontier frontier = player.getProductionFrontier();
    if (frontier != null) {
      boolean firstRule = true;
      for (final ProductionRule rule : frontier) {
        if (!firstRule) {
          sb.append(',');
        }
        firstRule = false;
        sb.append('{');
        appendJsonString(sb, "name", rule.getName());
        sb.append(',');
        sb.append("\"cost\":").append(rule.getCosts().getInt(pus));
        sb.append(',');
        // What the rule produces
        sb.append("\"results\":[");
        boolean firstResult = true;
        for (final Map.Entry<NamedAttachable, Integer> entry :
            rule.getResults().entrySet()) {
          if (!firstResult) {
            sb.append(',');
          }
          firstResult = false;
          sb.append('{');
          appendJsonString(sb, "name", entry.getKey().getName());
          sb.append(',');
          sb.append("\"quantity\":").append(entry.getValue());
          sb.append('}');
        }
        sb.append(']');
        sb.append('}');
      }
    }
    sb.append(']');

    // Units held by the player (purchased but not yet placed)
    sb.append(",\"unitsToPlace\":[");
    boolean firstHeld = true;
    for (final Unit u : player.getUnitCollection()) {
      if (!firstHeld) {
        sb.append(',');
      }
      firstHeld = false;
      sb.append('{');
      appendJsonString(sb, "type", u.getType().getName());
      sb.append(',');
      appendJsonString(sb, "owner", u.getOwner().getName());
      sb.append('}');
    }
    sb.append(']');

    sb.append('}');
    return sb.toString();
  }

  // ---------------------------------------------------------------------------
  // Action execution: Purchase
  // ---------------------------------------------------------------------------

  /**
   * Parses purchase actions from the server response and executes them.
   *
   * <p>Expected JSON format:
   * <pre>
   * {
   *   "purchases": [
   *     {"unitType": "infantry", "quantity": 3},
   *     {"unitType": "tank", "quantity": 1}
   *   ]
   * }
   * </pre>
   */
  private void executePurchaseActions(
      final String responseJson,
      final IPurchaseDelegate purchaseDelegate,
      final GameData data,
      final GamePlayer player,
      final boolean purchaseForBid) {

    final List<Map<String, String>> purchases = parseJsonArray(responseJson, "purchases");
    if (purchases.isEmpty()) {
      log.info("RLBot: no purchases requested by action server.");
      return;
    }

    final ProductionFrontier frontier = player.getProductionFrontier();
    if (frontier == null) {
      log.warn("RLBot: player has no production frontier, cannot purchase.");
      return;
    }

    // Build a lookup from rule name -> ProductionRule
    final Map<String, ProductionRule> rulesByName = new HashMap<>();
    // Also build a lookup from unit type name -> ProductionRule (for convenience)
    final Map<String, ProductionRule> rulesByUnitType = new HashMap<>();
    for (final ProductionRule rule : frontier) {
      rulesByName.put(rule.getName(), rule);
      for (final NamedAttachable result : rule.getResults().keySet()) {
        rulesByUnitType.put(result.getName().toLowerCase(), rule);
      }
    }

    final IntegerMap<ProductionRule> purchaseMap = new IntegerMap<>();
    for (final Map<String, String> purchase : purchases) {
      final String unitType = purchase.getOrDefault("unitType", "").toLowerCase();
      final int quantity;
      try {
        quantity = Integer.parseInt(purchase.getOrDefault("quantity", "0"));
      } catch (final NumberFormatException e) {
        log.warn("RLBot: invalid purchase quantity for unit type: " + unitType);
        continue;
      }
      if (quantity <= 0) {
        continue;
      }

      // Try to find the production rule
      ProductionRule rule = rulesByUnitType.get(unitType);
      if (rule == null) {
        // Try matching by rule name (e.g., "buyInfantry")
        final String buyName = "buy" + capitalize(unitType);
        rule = rulesByName.get(buyName);
      }
      if (rule == null) {
        log.warn("RLBot: no production rule found for unit type: " + unitType);
        continue;
      }
      purchaseMap.add(rule, quantity);
    }

    if (!purchaseMap.isEmpty()) {
      final String error = purchaseDelegate.purchase(purchaseMap);
      if (error != null) {
        log.warn("RLBot: purchase error: " + error);
      } else {
        log.info("RLBot: purchase successful: " + purchaseMap);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Action execution: Move
  // ---------------------------------------------------------------------------

  /**
   * Parses move actions from the server response and executes them.
   *
   * <p>Expected JSON format:
   * <pre>
   * {
   *   "moves": [
   *     {
   *       "unitTypes": ["infantry", "infantry", "tank"],
   *       "from": "Eastern Europe",
   *       "to": "Ukraine",
   *       "route": ["Eastern Europe", "Ukraine"]
   *     }
   *   ]
   * }
   * </pre>
   */
  private void executeMoveActions(
      final String responseJson,
      final IMoveDelegate moveDel,
      final GameData data,
      final GamePlayer player) {

    final List<Map<String, String>> moves = parseJsonArray(responseJson, "moves");
    if (moves.isEmpty()) {
      log.info("RLBot: no moves requested by action server.");
      return;
    }

    final GameMap gameMap = data.getMap();

    for (final Map<String, String> moveSpec : moves) {
      try {
        final String fromName = moveSpec.get("from");
        final String toName = moveSpec.get("to");
        final String unitTypesRaw = moveSpec.getOrDefault("unitTypes", "");
        final String routeRaw = moveSpec.getOrDefault("route", "");

        if (fromName == null || toName == null) {
          log.warn("RLBot: move missing 'from' or 'to' field, skipping.");
          continue;
        }

        final Territory from = gameMap.getTerritoryOrNull(fromName);
        final Territory to = gameMap.getTerritoryOrNull(toName);
        if (from == null || to == null) {
          log.warn("RLBot: unknown territory in move: " + fromName + " -> " + toName);
          continue;
        }

        // Parse the route (list of territory names)
        final List<Territory> routeTerritories = new ArrayList<>();
        if (!routeRaw.isEmpty()) {
          for (final String tName : parseSimpleJsonArray(routeRaw)) {
            final Territory t = gameMap.getTerritoryOrNull(tName.trim());
            if (t != null) {
              routeTerritories.add(t);
            }
          }
        }

        // Build the Route object
        final Route route;
        if (routeTerritories.size() >= 2) {
          route = new Route(routeTerritories);
        } else {
          route = new Route(from, to);
        }

        // Parse which unit types to move
        final List<String> unitTypeNames = parseSimpleJsonArray(unitTypesRaw);

        // Select units from the source territory matching the requested types
        final Collection<Unit> unitsInFrom = from.getUnitCollection().stream()
            .filter(u -> u.isOwnedBy(player))
            .collect(Collectors.toList());

        final List<Unit> unitsToMove = new ArrayList<>();
        final List<Unit> available = new ArrayList<>(unitsInFrom);

        for (final String typeName : unitTypeNames) {
          final String lowerType = typeName.trim().toLowerCase();
          // Find a matching unit in the available pool
          final Unit match = available.stream()
              .filter(u -> u.getType().getName().toLowerCase().equals(lowerType))
              .findFirst()
              .orElse(null);
          if (match != null) {
            unitsToMove.add(match);
            available.remove(match);
          } else {
            log.warn(
                "RLBot: could not find unit of type '"
                    + typeName
                    + "' in "
                    + fromName
                    + " for move.");
          }
        }

        if (unitsToMove.isEmpty()) {
          log.warn("RLBot: no valid units to move from " + fromName + " to " + toName);
          continue;
        }

        final MoveDescription moveDescription = new MoveDescription(unitsToMove, route);
        final Optional<String> error = moveDel.performMove(moveDescription);
        if (error.isPresent()) {
          log.warn("RLBot: move error (" + fromName + " -> " + toName + "): " + error.get());
        } else {
          log.info(
              "RLBot: moved "
                  + unitsToMove.size()
                  + " units from "
                  + fromName
                  + " to "
                  + toName);
        }
      } catch (final Exception e) {
        log.warn("RLBot: error executing move action.", e);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Action execution: Place
  // ---------------------------------------------------------------------------

  /**
   * Parses place actions from the server response and executes them.
   *
   * <p>Expected JSON format:
   * <pre>
   * {
   *   "placements": [
   *     {"territory": "Germany", "unitType": "infantry", "quantity": 3},
   *     {"territory": "Germany", "unitType": "tank", "quantity": 1}
   *   ]
   * }
   * </pre>
   */
  private void executePlaceActions(
      final String responseJson,
      final IAbstractPlaceDelegate placeDelegate,
      final GameData data,
      final GamePlayer player,
      final boolean placeForBid) {

    final List<Map<String, String>> placements = parseJsonArray(responseJson, "placements");
    if (placements.isEmpty()) {
      log.info("RLBot: no placements requested by action server.");
      return;
    }

    final GameMap gameMap = data.getMap();

    // Collect units held by the player that can be placed
    final List<Unit> heldUnits = new ArrayList<>(player.getUnitCollection());

    for (final Map<String, String> placement : placements) {
      final String territoryName = placement.get("territory");
      final String unitType = placement.getOrDefault("unitType", "").toLowerCase();
      final int quantity;
      try {
        quantity = Integer.parseInt(placement.getOrDefault("quantity", "0"));
      } catch (final NumberFormatException e) {
        log.warn("RLBot: invalid placement quantity.");
        continue;
      }

      if (territoryName == null || quantity <= 0) {
        continue;
      }

      final Territory territory = gameMap.getTerritoryOrNull(territoryName);
      if (territory == null) {
        log.warn("RLBot: unknown territory for placement: " + territoryName);
        continue;
      }

      // Find matching units from the player's held units
      final List<Unit> unitsToPlace = new ArrayList<>();
      for (int i = 0; i < quantity; i++) {
        final Unit match = heldUnits.stream()
            .filter(u -> u.getType().getName().toLowerCase().equals(unitType))
            .findFirst()
            .orElse(null);
        if (match != null) {
          unitsToPlace.add(match);
          heldUnits.remove(match);
        } else {
          log.warn("RLBot: no held unit of type '" + unitType + "' available to place.");
          break;
        }
      }

      if (!unitsToPlace.isEmpty()) {
        final BidMode bidMode = placeForBid ? BidMode.BID : BidMode.NOT_BID;
        final Optional<String> error = placeDelegate.placeUnits(unitsToPlace, territory, bidMode);
        if (error.isPresent()) {
          log.warn(
              "RLBot: placement error in "
                  + territoryName
                  + ": "
                  + error.get());
        } else {
          log.info(
              "RLBot: placed "
                  + unitsToPlace.size()
                  + " "
                  + unitType
                  + " in "
                  + territoryName);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // HTTP communication
  // ---------------------------------------------------------------------------

  /**
   * Sends a POST request with the given JSON body to the action server and returns the response
   * body, or null if the request fails.
   */
  @Nullable
  private String postToActionServer(final String jsonBody) {
    HttpURLConnection conn = null;
    try {
      final URL url = new URL(ACTION_SERVER_URL);
      conn = (HttpURLConnection) url.openConnection();
      conn.setRequestMethod("POST");
      conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
      conn.setConnectTimeout(HTTP_TIMEOUT_MS);
      conn.setReadTimeout(HTTP_TIMEOUT_MS);
      conn.setDoOutput(true);

      try (final OutputStream os = conn.getOutputStream()) {
        os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
        os.flush();
      }

      final int responseCode = conn.getResponseCode();
      if (responseCode != 200) {
        log.warn("RLBot: action server returned HTTP " + responseCode);
        return null;
      }

      try (final BufferedReader reader =
          new BufferedReader(
              new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
        final StringBuilder response = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
          response.append(line);
        }
        return response.toString();
      }
    } catch (final Exception e) {
      log.warn("RLBot: failed to contact action server at " + ACTION_SERVER_URL + ": " + e.getMessage());
      return null;
    } finally {
      if (conn != null) {
        conn.disconnect();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Minimal JSON helpers (no external library dependency)
  // ---------------------------------------------------------------------------

  /**
   * Parses a JSON array of objects under the given key from a JSON string.
   * Returns a list of maps where each map represents one JSON object.
   *
   * <p>This is a minimal parser sufficient for the simple flat-object arrays
   * the action server returns. It does not handle nested objects or escaped quotes
   * beyond basic cases.
   */
  private List<Map<String, String>> parseJsonArray(final String json, final String key) {
    final List<Map<String, String>> result = new ArrayList<>();
    final String searchKey = "\"" + key + "\"";
    final int keyIndex = json.indexOf(searchKey);
    if (keyIndex < 0) {
      return result;
    }

    // Find the opening bracket of the array
    final int arrayStart = json.indexOf('[', keyIndex);
    if (arrayStart < 0) {
      return result;
    }

    // Find the matching closing bracket
    int depth = 0;
    int arrayEnd = -1;
    for (int i = arrayStart; i < json.length(); i++) {
      final char c = json.charAt(i);
      if (c == '[') {
        depth++;
      } else if (c == ']') {
        depth--;
        if (depth == 0) {
          arrayEnd = i;
          break;
        }
      }
    }
    if (arrayEnd < 0) {
      return result;
    }

    final String arrayContent = json.substring(arrayStart + 1, arrayEnd).trim();
    if (arrayContent.isEmpty()) {
      return result;
    }

    // Split by top-level objects (delimited by { ... })
    int objDepth = 0;
    int objStart = -1;
    for (int i = 0; i < arrayContent.length(); i++) {
      final char c = arrayContent.charAt(i);
      if (c == '{') {
        if (objDepth == 0) {
          objStart = i;
        }
        objDepth++;
      } else if (c == '}') {
        objDepth--;
        if (objDepth == 0 && objStart >= 0) {
          final String objStr = arrayContent.substring(objStart + 1, i).trim();
          result.add(parseSimpleJsonObject(objStr));
          objStart = -1;
        }
      }
    }

    return result;
  }

  /**
   * Parses a flat JSON object (no nesting) into a string-to-string map.
   * Handles both string and numeric values by converting everything to strings.
   */
  private Map<String, String> parseSimpleJsonObject(final String objContent) {
    final Map<String, String> map = new HashMap<>();
    // Match patterns like "key": "value" or "key": 123 or "key": ["a","b"]
    int i = 0;
    while (i < objContent.length()) {
      // Find key
      final int keyStart = objContent.indexOf('"', i);
      if (keyStart < 0) {
        break;
      }
      final int keyEnd = objContent.indexOf('"', keyStart + 1);
      if (keyEnd < 0) {
        break;
      }
      final String currentKey = objContent.substring(keyStart + 1, keyEnd);

      // Find colon
      final int colon = objContent.indexOf(':', keyEnd + 1);
      if (colon < 0) {
        break;
      }

      // Find value start (skip whitespace)
      int valStart = colon + 1;
      while (valStart < objContent.length() && objContent.charAt(valStart) == ' ') {
        valStart++;
      }
      if (valStart >= objContent.length()) {
        break;
      }

      final char firstChar = objContent.charAt(valStart);
      String value;
      int valEnd;

      if (firstChar == '"') {
        // String value
        valEnd = objContent.indexOf('"', valStart + 1);
        if (valEnd < 0) {
          break;
        }
        value = objContent.substring(valStart + 1, valEnd);
        i = valEnd + 1;
      } else if (firstChar == '[') {
        // Array value -- store as raw string
        int bracketDepth = 0;
        valEnd = valStart;
        for (int j = valStart; j < objContent.length(); j++) {
          if (objContent.charAt(j) == '[') {
            bracketDepth++;
          } else if (objContent.charAt(j) == ']') {
            bracketDepth--;
            if (bracketDepth == 0) {
              valEnd = j;
              break;
            }
          }
        }
        value = objContent.substring(valStart, valEnd + 1);
        i = valEnd + 1;
      } else {
        // Numeric or boolean value
        valEnd = valStart;
        while (valEnd < objContent.length()
            && objContent.charAt(valEnd) != ','
            && objContent.charAt(valEnd) != '}') {
          valEnd++;
        }
        value = objContent.substring(valStart, valEnd).trim();
        i = valEnd;
      }

      map.put(currentKey, value);
    }
    return map;
  }

  /**
   * Parses a simple JSON array of strings, e.g. ["foo", "bar", "baz"].
   * Also handles raw comma-separated values without brackets.
   */
  private List<String> parseSimpleJsonArray(final String raw) {
    final List<String> result = new ArrayList<>();
    String content = raw.trim();
    if (content.startsWith("[")) {
      content = content.substring(1);
    }
    if (content.endsWith("]")) {
      content = content.substring(0, content.length() - 1);
    }
    if (content.isEmpty()) {
      return result;
    }
    for (final String part : content.split(",")) {
      String val = part.trim();
      if (val.startsWith("\"") && val.endsWith("\"")) {
        val = val.substring(1, val.length() - 1);
      }
      if (!val.isEmpty()) {
        result.add(val);
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // JSON output helpers
  // ---------------------------------------------------------------------------

  private static void appendJsonString(
      final StringBuilder sb, final String key, final String value) {
    sb.append('"').append(escapeJson(key)).append("\":\"").append(escapeJson(value)).append('"');
  }

  private static String escapeJson(final String s) {
    if (s == null) {
      return "";
    }
    return s.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t");
  }

  private static String capitalize(final String s) {
    if (s == null || s.isEmpty()) {
      return s;
    }
    return s.substring(0, 1).toUpperCase() + s.substring(1);
  }
}
