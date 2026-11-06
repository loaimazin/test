#include "include/solitaire_auto.h"
#include "include/ghost_sim.h"
#include "include/license_security.h"
#include "include/il2cpp_exports.h"
#include "include/solitaire_liveness_policy.h"

#include <dlfcn.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <queue>
#include <unistd.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Jawaker 28.3.16 metadata and ARM64 function entries. Keep the V6.1 planner
// unchanged; only its game-version adapter is relocated here.
constexpr uintptr_t RVA_SOL_UI_GET_INSTANCE  = 0x04D1B4B8UL;
constexpr uintptr_t RVA_SOL_GET_STACK_CARDS  = 0x04A46B80UL;
constexpr uintptr_t RVA_SOL_MOVE_CARD        = 0x04A46C74UL;
constexpr uintptr_t RVA_SOL_TALON_CLICKED    = 0x04A45644UL;
// Native SolitaireLocalGs::SyncPlayerCardsMapping(). IDA confirms that this
// reads the authoritative id->value arrays from GameStore and stores them at
// LocalGs + 0xD8. Calling only this narrow sync avoids touching gameplay state.
constexpr uintptr_t RVA_SOL_SYNC_PLAYER_CARDS_MAPPING = 0x04A4AAB4UL;
constexpr uintptr_t RVA_SOL_RAY_MOVE_CARD     = 0x04D16610UL;
// Solitaire.TalonClicked() is static in both game versions. 0x04D1A560 is
// SolitaireTalonPile.OnTalonClicked(), an instance UI callback; calling it
// with a null `this` is unsafe when drawing from stock.
constexpr uintptr_t RVA_SOL_UI_TALON_CLICK    = 0x04D11ADCUL;
// static int Jawaker.Games.Core.GameStore.GetMyIndex(). LocalGs.Stacks holds
// every player's piles ("p_0_...", "p_1_..."); this index names ours.
constexpr uintptr_t RVA_GAMESTORE_GET_MY_INDEX = 0x04B26EE8UL;
// UnityEngine.Application.get/set_targetFrameRate icall wrappers (W0 only, no
// MethodInfo). Jawaker's AppInit.InitAppConfig pins the app to 60 FPS with
// vSyncCount=0, so every frame-quantized gate below costs 16.7 ms per action.
constexpr uintptr_t RVA_APP_GET_TARGET_FRAME_RATE = 0x09F55930UL;
constexpr uintptr_t RVA_APP_SET_TARGET_FRAME_RATE = 0x09F55958UL;
constexpr size_t OFF_SOL_UI_LOCAL_GS          = 0x80;
constexpr size_t OFF_SOL_LOCAL_SYNC_APPLIED   = 0x19;
constexpr size_t OFF_SOL_IS_UPDATING_UI       = 0x1A;
constexpr size_t OFF_SOL_CURRENT_SEQ           = 0x28;
constexpr size_t OFF_SOL_PENDING_MOVES         = 0x30;
constexpr size_t OFF_SOL_MOVE_SEQ              = 0x18;
constexpr size_t OFF_JARRAY_VALUES            = 0x58;
constexpr size_t OFF_LIST_ITEMS               = 0x10;
constexpr size_t OFF_LIST_SIZE                = 0x18;
constexpr size_t OFF_ARRAY_ITEMS              = 0x20;
constexpr size_t OFF_JVALUE_VALUE             = 0x38;
constexpr size_t OFF_JOBJECT_PROPERTIES        = 0x58;
constexpr size_t OFF_COLLECTION_ITEMS          = 0x10;
constexpr size_t OFF_JPROPERTY_CONTENT         = 0x58;
constexpr size_t OFF_JPROPERTY_NAME            = 0x60;
constexpr size_t OFF_JPROPERTYLIST_TOKEN       = 0x10;
constexpr size_t OFF_SOL_PLAYER_CARDS_MAPPING  = 0xD8;
constexpr size_t OFF_SOL_START_TIME            = 0x48;
constexpr size_t OFF_SOL_REVEALED_IDS          = 0xE0;
constexpr size_t OFF_SOL_CARDS_PER_TALON_FLIP = 0xF8;
constexpr size_t OFF_SOL_STACKS                = 0x110;
constexpr size_t OFF_SOL_HS_STACKS             = 0x118;
constexpr size_t OFF_HASHSET_SLOTS             = 0x18;
constexpr size_t OFF_HASHSET_LAST_INDEX        = 0x24;
// Jawaker.Games.Core.GameStore static-field layout in V28.3.16. `gs` is the
// raw JObject received by SetGS, before SolitaireLocalGs masks face-down card
// values for the local player's rendered state.
constexpr size_t OFF_GAMESTORE_STATIC_GS        = 0x38;

// KING LIVENESS V21 keeps exact logical/physical acknowledgement authoritative, but
// stops adding a second artificial pause once Jawaker has already published the
// expected state. A failed transition still raises the adaptive round-local
// backoff, so the one-millisecond floor is earned rather than blindly assumed.
constexpr uint64_t SOL_CARD_ACTION_SETTLE_MS      = 1;
constexpr uint64_t SOL_TALON_ACTION_SETTLE_MS     = 1;
constexpr uint64_t SOL_HUMAN_CARD_ACTION_MS       = 320;
constexpr uint64_t SOL_HUMAN_TALON_ACTION_MS      = 180;
// 700 ms is a diagnostic/soft boundary only. DropCard can use an optimistic
// LocalGs update or wait for server reconciliation; the native call itself has
// no success return and there is no proven 700 ms server upper bound.
constexpr uint64_t SOL_ACTION_RESULT_TIMEOUT_MS   = 700;
constexpr uint64_t SOL_ACTION_HARD_TIMEOUT_MS     = 2200;
constexpr int SOL_OPTIMISTIC_PIPELINE_MAX =
    solitaire_liveness_policy::kOptimisticPipelineMax;
// Server mode (IsLocalSolitaireEnabled == false): DropCard/TalonClicked only
// send the action and the board changes when the server answers, ~260 ms
// later. Measured on 28.3.16 practice tables with a Frida auto-player, the
// server applies actions sent before it has answered, in order, with no
// rejections: 1 in flight ~3.8 moves/s, 3 in flight ~8-10 moves/s. Only
// committed-plan steps whose resulting board is fully known are sent ahead.
constexpr int SOL_SERVER_PIPELINE_MAX = 3;
// After discarding queued steps, the server may still apply ones already sent.
// Wait out one loaded reply time before planning from the observed board.
constexpr uint64_t SOL_PIPELINE_DRAIN_MS = 600;
constexpr uint64_t SOL_PIPELINE_RETRY_MS          = 8;
// Match the reference engine's verified-flight shape: after the exact logical
// result appears, require a short stable/non-animating observation before a
// different card is dispatched. This is verification, not a blind move delay.
// One full 60-Hz frame after exact physical ACK is enough to detect a late
// queue/UI change; a second unconditional frame made every SAFE X3 action
// visibly slower on the captured phone even with a clean 170-ms Unity move.
constexpr uint64_t SOL_POST_ACK_STABLE_MS          = 8;
constexpr uint64_t SOL_UI_UPDATE_WATCHDOG_MS      = 240;
// Phone FAST profiles: Unity keeps `isUpdatingUI` raised for most of the visual
// Talon/card animation even though LocalGs has already committed the move. The
// exact-state acknowledgement below remains authoritative, so a shorter bounded
// watchdog removes the visible "heavy flip" pause without blindly double-clicking.
// The PC emulator and BALANCED profiles keep the conservative 240 ms gate.
constexpr uint64_t SOL_PHONE_TALON_UI_WATCHDOG_MS = 36;
constexpr uint64_t SOL_PHONE_CARD_UI_WATCHDOG_MS  = 52;
constexpr uint64_t SOL_CARD_VIEW_RETRY_MS          = 12;
constexpr uint64_t SOL_CARD_VIEW_REFRESH_MS        = 120;
constexpr uint64_t SOL_MAX_PACING_BACKOFF_MS      = 180;
constexpr uint64_t SOL_PACING_BACKOFF_STEP_MS     = 28;
constexpr uint64_t SOL_PACING_RECOVERY_STEP_MS    = 28;
constexpr uint64_t SOL_EXHAUSTED_RETRY_MS         = 260;
// A live board must never require the player to tap Talon to wake the solver.
// After this bounded proof window, a Talon edge is eligible only when a
// complete virtual stock scan proves that it exposes an unvisited legal exit.
// The ordinary pending-flight ACK still serializes the physical tap.
constexpr uint64_t SOL_TALON_LIVENESS_MS          = 90;
constexpr uint64_t SOL_MAPPING_SYNC_RETRY_MS       = 240;
// While Auto Solitaire runs, ask Unity for up to 120 FPS. Android still caps
// this at the panel refresh rate, so a 60 Hz phone is unaffected while a
// 90/120 Hz phone ticks the planner and observes animation ends more often.
// The game's original target is restored when Auto is switched off.
constexpr int SOL_BOOST_TARGET_FRAME_RATE          = 120;
constexpr uint64_t SOL_FRAME_RATE_RECHECK_MS       = 1000;
constexpr int SOL_NORMAL_SEARCH_FRAMES             = 1;
constexpr int SOL_ENDGAME_SEARCH_FRAMES             = 2;
constexpr int SOL_MAX_RECOVERY_ESCALATION          = 3;
constexpr int SOL_MAPPING_SYNC_MAX_ATTEMPTS        = 12;
constexpr int SOL_MAPPING_SYNC_RECOVERY_BURSTS     = 3;
// Emergency look-ahead is a last-resort bridge, not a second free-running
// solver.  Bound cosmetic preparations across the whole no-progress epoch so
// a sequence of individually new Tableau layouts cannot become a long visual
// shuffle loop. Normal planner lines are unaffected by this limit.
constexpr int SOL_MAX_EMERGENCY_PREPARATIONS       = 6;
// Speculative recovery bridges are not progress. Keep a small per-epoch
// allowance, then spend bounded search time proving a real milestone instead
// of physically permuting the same blocked board indefinitely.
constexpr int SOL_MAX_RECOVERY_PREPARATIONS        = 6;
constexpr int SOL_RECOVERY_PROOF_ONLY_FRAMES       = 72;
constexpr size_t SOL_STRUCTURAL_CACHE_LIMIT        = 65536;
// Victory/RAPTOR switches to proof-first play before the old seven-hidden-card
// boundary.  A deal with most cards known can already contain the decisive
// dependency fork, and waiting until the last few cards is too late to undo a
// superficially attractive reveal.  Foundation progress is an independent
// endgame signal because many fast deals expose the Tableau early.
constexpr int SOL_VICTORY_ENDGAME_HIDDEN            = 12;
constexpr int SOL_VICTORY_ENDGAME_FOUNDATION        = 20;
// Consume the server/animation flight window on the worker instead of making
// the next board wait for fresh Unity-thread search. The worker is cancelled
// between short slices as soon as the physical prediction becomes obsolete.
constexpr uint64_t SOL_RAPTOR_PREFETCH_MS           = 190;
constexpr int SOL_RAPTOR_PREFETCH_MAX_SLICES        = 64;
// The first three full Talon recycles are free in this match type. A charged
// pass is less attractive, but must never be forbidden if needed to win.
constexpr int SOL_FREE_TALON_RECYCLES              = 3;
// Every Talon click still has a small tempo price.  More importantly, the
// fourth and later complete recycle costs the player 20 real points in this
// ruleset.  Treat that paid pass as a first-class route objective instead of a
// weak tie-break: it remains legal when it is the only winning bridge, but an
// equally safe line that stays inside the three free passes wins decisively.
constexpr int SOL_TALON_STEP_SEARCH_COST           = 900;
constexpr int SOL_CHARGED_RECYCLE_SEARCH_COST      = 180000;
constexpr int SOL_CHARGED_RECYCLE_SCORE_DELTA      = 20;

using StringNewFn      = void* (*)(const char*);
using ObjectToStringFn = void* (*)(void*, void**);
using ClassNameFn      = const char* (*)(void*);
using DomainGetFn      = void* (*)();
using DomainAssembliesFn = const void** (*)(void*, size_t*);
using AssemblyImageFn  = void* (*)(const void*);
using ImageNameFn      = const char* (*)(void*);
using ClassFromNameFn  = void* (*)(void*, const char*, const char*);
using StaticDataFn     = void* (*)(void*);
using RayMoveFn        = void (*)(void*, void*, int32_t, void*, void*);
using GetUiFn          = void* (*)(void*);
using GetStackCardsFn  = void* (*)(void*, void*, void*);
using MoveCardFn       = bool  (*)(void*, int32_t, void*, void*, void*);
using TalonClickedFn   = bool  (*)(void*, void*);
using TalonUiFn        = void  (*)(void*);
using SyncPlayerCardsMappingFn = bool (*)(void*);
using GetTargetFrameRateFn = int32_t (*)();
using SetTargetFrameRateFn = void (*)(int32_t);
using GetMyIndexFn = int32_t (*)(void*);

StringNewFn     g_stringNew = nullptr;
ObjectToStringFn g_objectToString = nullptr;
ClassNameFn      g_classGetName = nullptr;
DomainGetFn      g_domainGet = nullptr;
DomainAssembliesFn g_domainAssemblies = nullptr;
AssemblyImageFn  g_assemblyImage = nullptr;
ImageNameFn      g_imageName = nullptr;
ClassFromNameFn  g_classFromName = nullptr;
StaticDataFn     g_staticData = nullptr;
RayMoveFn        g_rayMove = nullptr;
GetUiFn         g_getUi = nullptr;
GetStackCardsFn g_getStackCards = nullptr;
MoveCardFn      g_moveCard = nullptr;
TalonClickedFn  g_talonClicked = nullptr;
TalonUiFn       g_talonUi = nullptr;
SyncPlayerCardsMappingFn g_syncPlayerCardsMapping = nullptr;
GetTargetFrameRateFn g_getTargetFrameRate = nullptr;
SetTargetFrameRateFn g_setTargetFrameRate = nullptr;
GetMyIndexFn    g_getMyIndex = nullptr;
int             g_myPlayerIndex = -1;         // refreshed once a second
uint64_t        g_myPlayerIndexAt = 0;
int             g_savedTargetFrameRate = 0;   // 0 = not boosted
uint64_t        g_lastFrameRateCheckMs = 0;
void*           g_ui = nullptr;
void*           g_rayController = nullptr;
void*           g_cardsController = nullptr;
void*           g_gameStoreStatics = nullptr;
void*           g_lastLocalGs = nullptr;
int64_t         g_lastRoundStartTime = 0;
// A 320k-node proof frontier is intentionally allowed to grow large during a
// hard deal. Clearing its C++ containers at the next deal releases ownership,
// but Android's allocator can keep those pages in the process until the app is
// restarted. V18 performs one allocator purge at a proven round boundary,
// after the anticipatory worker has dropped every old-round object.
std::atomic<bool> g_roundMemoryReleasePending{false};
uint64_t g_roundMemoryReleaseRequestedAt = 0;
uint64_t g_lastRoundMemoryReleaseAt = 0;

std::atomic<bool> g_enabled{false};
// UI clicks arrive on Android's UI thread while the planner owns the Unity
// thread gate. Record the intent atomically; apply its state reset on Unity.
std::atomic<bool> g_enableResetPending{false};
uint64_t g_lastMoveMs = 0;
uint64_t g_lastTalonMs = 0;
bool     g_lastActionWasTalon = false;
// Starts in the fast profile. A physical action that fails to materialize
// raises this bounded delay automatically; successful actions decay it back to
// zero. This preserves speed on healthy devices without turning a slow Unity
// animation into dropped moves or a stalled committed plan.
uint64_t g_pacingBackoffMs = 0;
// The loader selects a device-specific search slice after libengine is loaded.
// These knobs affect virtual planning work only; physical card/talon dispatch
// remains protected by the exact-state acknowledgement and adaptive backoff.
// That separation lets PC emulators spend more CPU per frame without making a
// slower phone drop Unity input or execute a stale committed-plan move.
std::atomic<int> g_searchSliceNodes{6400};
std::atomic<int> g_searchSliceMs{4};
std::atomic<int> g_searchArenaNodes{320000};
std::atomic<int> g_performanceProfile{2};
// Runtime overlay choice. FAST preserves the proven phone profile byte-for-
// byte in the hot path; HUMAN adds only a natural dispatch interval and never
// changes search quality, move ordering, or exact-state validation.
std::atomic<int> g_speedMode{1};

static bool phoneFastUiProfile() {
    const int profile = g_performanceProfile.load(std::memory_order_relaxed);
    return profile == 0 || profile == 2;
}

static uint64_t effectiveUiWatchdogMs() {
    if (!phoneFastUiProfile()) return SOL_UI_UPDATE_WATCHDOG_MS;

    // Back off automatically on a device/round where Unity rejected an action.
    // Two misses restore almost the original 240 ms window; successful exact
    // acknowledgements decay this penalty again, so healthy rounds stay fast.
    const uint64_t fastBase = g_lastActionWasTalon
        ? SOL_PHONE_TALON_UI_WATCHDOG_MS
        : SOL_PHONE_CARD_UI_WATCHDOG_MS;
    return std::min(SOL_UI_UPDATE_WATCHDOG_MS,
                    fastBase + std::min<uint64_t>(g_pacingBackoffMs, 95));
}

static uint64_t pacingRecoveryStepMs() {
    // The phone fast path should not remain slow for several successful moves
    // after one transient animation miss. PC/default behavior is unchanged.
    return phoneFastUiProfile() ? 45 : SOL_PACING_RECOVERY_STEP_MS;
}

static bool humanSpeedMode() {
    return g_speedMode.load(std::memory_order_relaxed) == 0;
}
int      g_lastMoveId = -1;
uint64_t g_lastMoveAttemptMs = 0;
std::string g_lastMoveFrom;
std::string g_lastMoveTo;
enum class DispatchOutcome : uint8_t { Rejected, Deferred, Submitted };
static DispatchOutcome g_lastDispatchOutcome = DispatchOutcome::Rejected;
static bool g_dispatchDeferredThisTick = false;
static uint64_t g_dispatchRetryAtMs = 0;
static std::string g_dispatchRetryPhysicalKey;
static int64_t g_lastSubmittedSequence = -1;
static int64_t g_lastSubmittedSequenceBefore = -1;
static int g_lastSubmittedPendingCount = 0;
static bool g_lastSubmittedWasQueued = false;
static int g_lastSubmittedFoundationRescueCardId = -1;
static uint64_t g_cardViewMissSinceMs = 0;
static uint64_t g_lastCardViewRefreshMs = 0;
static int g_cardViewMissId = -1;
struct MoveRecord {
    int cardId;
    std::string from;
    std::string to;
    uint64_t at;
};
std::vector<MoveRecord> g_moveHistory;
struct PositionRecord {
    std::string tableau;
    uint64_t at;
};
std::vector<PositionRecord> g_positionHistory;
std::vector<PositionRecord> g_solverStateHistory;
int      g_lastRecycleScore = -1000000;
bool     g_rescueMode = false;
int      g_rrSource = 0;
uint64_t g_lastDiagMs = 0;
uint64_t g_lastCardDiagMs = 0;
uint64_t g_lastStallDiagMs = 0;
bool     g_loggedStackKeys = false;
bool     g_loggedStackStorage = false;
bool     g_loggedSnapshot = false;
bool     g_loggedPerfectSnapshot = false;
bool     g_loggedMappingSnapshot = false;
bool     g_loggedHiddenStateSnapshot = false;
std::string g_stackPrefix = "p_0_";
std::atomic_flag g_solitaireAccess = ATOMIC_FLAG_INIT;
std::atomic<int> g_accessOwnerTid{0};
std::atomic<uint64_t> g_accessAcquiredMs{0};
std::atomic<int> g_accessStage{0};
int      g_actualRecycleCount = 0;   // whole-round count for score-aware search
int      g_stagnantRecycleCount = 0; // resets after a productive card action
int      g_cardsPerTalonFlip = 3;
// Values learned during this physical round must survive transient empty
// PlayerCardsMapping snapshots and controller animation gaps.  A value is
// keyed by the game's stable card id; -1 means that card is still opaque.
std::array<int, 52> g_roundCardValues = [] {
    std::array<int, 52> values{};
    values.fill(-1);
    return values;
}();
// A Foundation pullback may reopen a blocked Tableau, but the same physical
// card must not be pulled twice before a real reveal/new Foundation high-water
// mark. This directly prevents Foundation <-> Tableau recovery ping-pong.
std::array<int, 52> g_foundationRescueEpoch = [] {
    std::array<int, 52> epochs{};
    epochs.fill(-1);
    return epochs;
}();
int      g_lastRoundKnownCount = 0;
uint64_t g_lastPerfectInfoDiagMs = 0;
uint64_t g_lastMappingSyncMs = 0;
int      g_mappingSyncAttempts = 0;
int      g_mappingRecoveryBursts = 0;
bool     g_mappingValidated = false;
// Real Tableau visibility, tracked across snapshots (see
// normalizeTableauVisibility): the card ids of each column at the last full
// snapshot and how many of its bottom cards are face down for this player.
struct ColumnVisibility {
    std::vector<int> ids;
    int hidden = 0;
};
std::array<ColumnVisibility, 7> g_columnVisibility;
bool     g_columnVisibilityValid = false;
int      g_leakMaskedCards = 0;

static void resetExpertPlanner(bool releaseCapacity = false);

#if GOAT_SOLITAIRE_TRACE
static void traceSolitaireToFile(const char* format, ...) {
    // Samsung's logcat buffer is empty on the test phone. Keep this diagnostic
    // path opt-in, bounded, and limited to decisions/state changes; per-node
    // search messages must never do file I/O on UnityMain.
    const char* markers[] = {
        "init ", "auto=", "captured UI", "new solitaire", "waiting ",
        "[GATE_RECOVER]", "[TICK_EXCEPTION]", "using LocalGs",
        "[ACTION", "[STALL]", "expert selected", "expert committed",
        "expert stalled", "expert talon", "expert foundation", "emergency",
        "[RECOVERY", "[PLAN_GUARD]", "[ENDGAME]", "[VICTORY_ORACLE]", "[DIAG]",
        "ray-move", "card-view miss", "perfect ", "[MAPPING_RECOVERY]"
    };
    bool wanted = false;
    for (const char* marker : markers) {
        if (std::strstr(format, marker)) { wanted = true; break; }
    }
    if (!wanted) return;
    static std::mutex traceMutex;
    // Keep the current and previous segment instead of silently disabling
    // diagnostics after 512 KiB in a long multi-game session. Use a new path
    // so the captured V22 evidence remains untouched.
    constexpr size_t segmentBytes = 4 * 1024 * 1024;
    constexpr const char* tracePath =
        "/sdcard/Android/data/com.jawaker.engine/files/solitaire_trace_live.log";
    constexpr const char* previousPath =
        "/sdcard/Android/data/com.jawaker.engine/files/solitaire_trace_live.previous.log";
    static size_t written = 0;
    static bool sizeInitialized = false;
    std::lock_guard<std::mutex> lock(traceMutex);
    char line[2048];
    va_list args;
    va_start(args, format);
    const int size = std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (size <= 0) return;
    // A stalled unchanged board can produce the same event on several Unity
    // ticks. Keep periodic proof of liveness without spending the file budget
    // or synchronously opening the file for every duplicate.
    static std::string lastLine;
    static uint64_t lastLineAt = 0;
    const uint64_t stamp = nowMs();
    if (lastLine == line && stamp - lastLineAt < 5000) return;
    lastLine = line;
    lastLineAt = stamp;
    FILE* file = std::fopen(tracePath, "ab");
    if (!file) return;
    if (!sizeInitialized) {
        if (std::fseek(file, 0, SEEK_END) == 0) {
            const long bytes = std::ftell(file);
            if (bytes >= 0) written = (size_t)bytes;
        }
        sizeInitialized = true;
    }
    if (written >= segmentBytes) {
        std::fclose(file);
        // Android/POSIX rename replaces only our previous diagnostic segment.
        // On failure preserve the current file rather than truncating it.
        if (std::rename(tracePath, previousPath) != 0) return;
        written = 0;
        file = std::fopen(tracePath, "ab");
        if (!file) return;
    }
    const int count = std::min(size, (int)sizeof(line) - 1);
    const int prefixBytes = std::fprintf(file, "%llu ", (unsigned long long)stamp);
    if (prefixBytes > 0) written += (size_t)prefixBytes;
    written += std::fwrite(line, 1, (size_t)count, file);
    written += std::fwrite("\n", 1, 1, file);
    std::fclose(file);
}
#define SOL_LOG(...) do { \
    __android_log_print(ANDROID_LOG_INFO, "GOAT_SOL", __VA_ARGS__); \
    traceSolitaireToFile(__VA_ARGS__); \
} while (0)
#else
// The diagnostic build retains Android/file logs. Ordinary play must not
// format or emit strings on the Unity decision path.
#define SOL_LOG(...) ((void)0)
#endif
// The opt-in file trace records decisions, not per-frame/per-node chatter.
#define SOL_TRACE(...) ((void)0)

struct Card {
    int id = -1;
    int value = -1;
    // The serialized stack hides a face-down value as -1. CardsController
    // still owns the real value, so planning can remember it without making
    // the physical card movable before the game reveals it.
    int knownValue = -1;
    std::string color;
    // Search-only marker.  When a legal virtual move exposes a card whose
    // value is not currently available from IL2CPP, the reveal itself is
    // still real progress.  Keep the card immovable until the next physical
    // snapshot supplies its value, but do not incorrectly treat it as hidden.
    bool virtuallyRevealed = false;
    bool valid() const { return id >= 0; }
    bool faceUp() const { return id >= 0 && value >= 0 && value < 52; }
    bool known() const { return faceUp() || (knownValue >= 0 && knownValue < 52); }
    bool revealedForPlanning() const { return faceUp() || virtuallyRevealed; }
};

static bool tryEnterSolitaire() {
    if (g_solitaireAccess.test_and_set(std::memory_order_acquire)) {
        const int tid = gettid();
        const int owner = g_accessOwnerTid.load(std::memory_order_acquire);
        const uint64_t acquired = g_accessAcquiredMs.load(std::memory_order_acquire);
        const uint64_t age = acquired ? nowMs() - acquired : 0;
        const int stage = g_accessStage.load(std::memory_order_acquire);
        // Every Solitaire hook here runs on UnityMain. A search slice has a
        // hard 3-15 ms deadline, so a same-thread search gate held for 1.5 s
        // cannot be a still-running normal slice. Do not reclaim a gate while
        // a native move is in flight or while another thread owns it.
        if (owner != tid || age < 1500 || stage != 4) return false;
        SOL_LOG("[GATE_RECOVER] owner=%d age=%llu stage=%d",
                owner, (unsigned long long)age, stage);
        g_enableResetPending.store(true, std::memory_order_release);
        g_accessStage.store(0, std::memory_order_release);
        g_accessAcquiredMs.store(0, std::memory_order_release);
        g_accessOwnerTid.store(0, std::memory_order_release);
        g_solitaireAccess.clear(std::memory_order_release);
        if (g_solitaireAccess.test_and_set(std::memory_order_acquire)) return false;
    }
    g_accessOwnerTid.store(gettid(), std::memory_order_release);
    g_accessAcquiredMs.store(nowMs(), std::memory_order_release);
    g_accessStage.store(1, std::memory_order_release);
    return true;
}

struct SolitaireAccessGuard {
    ~SolitaireAccessGuard() {
        const int stage = g_accessStage.load(std::memory_order_acquire);
        if (stage >= 4)
            SOL_TRACE("[GATE_EXIT] tid=%d stage=%d", gettid(), stage);
        g_accessStage.store(0, std::memory_order_release);
        g_accessAcquiredMs.store(0, std::memory_order_release);
        g_accessOwnerTid.store(0, std::memory_order_release);
        g_solitaireAccess.clear(std::memory_order_release);
    }
};

static void resetRoundState(bool resetControllers, bool releaseCapacity = false) {
    g_lastMoveMs = 0;
    g_lastTalonMs = 0;
    g_lastActionWasTalon = false;
    g_pacingBackoffMs = 0;
    g_lastMoveId = -1;
    g_lastMoveAttemptMs = 0;
    g_lastMoveFrom.clear();
    g_lastMoveTo.clear();
    if (releaseCapacity) {
        std::vector<MoveRecord>().swap(g_moveHistory);
        std::vector<PositionRecord>().swap(g_positionHistory);
        std::vector<PositionRecord>().swap(g_solverStateHistory);
    } else {
        g_moveHistory.clear();
        g_positionHistory.clear();
        g_solverStateHistory.clear();
    }
    g_lastRecycleScore = -1000000;
    g_rescueMode = false;
    g_rrSource = 0;
    g_actualRecycleCount = 0;
    g_stagnantRecycleCount = 0;
    g_cardsPerTalonFlip = 3;
    g_roundCardValues.fill(-1);
    g_foundationRescueEpoch.fill(-1);
    g_lastRoundKnownCount = 0;
    g_lastPerfectInfoDiagMs = 0;
    g_lastMappingSyncMs = 0;
    g_mappingSyncAttempts = 0;
    g_mappingRecoveryBursts = 0;
    g_mappingValidated = false;
    for (ColumnVisibility& column : g_columnVisibility) {
        column.ids.clear();
        column.hidden = 0;
    }
    g_columnVisibilityValid = false;
    g_leakMaskedCards = 0;
    g_myPlayerIndex = -1;
    g_myPlayerIndexAt = 0;
    g_lastStallDiagMs = 0;
    g_loggedStackKeys = false;
    g_loggedStackStorage = false;
    g_loggedSnapshot = false;
    g_loggedPerfectSnapshot = false;
    g_loggedMappingSnapshot = false;
    g_loggedHiddenStateSnapshot = false;
    g_stackPrefix = "p_0_";
    resetExpertPlanner(releaseCapacity);
    if (releaseCapacity) {
        g_roundMemoryReleaseRequestedAt = nowMs();
        g_roundMemoryReleasePending.store(true, std::memory_order_release);
    }
    if (resetControllers) {
        g_rayController = nullptr;
        g_cardsController = nullptr;
    }
}

static void* listItem(void* list, int index) {
    if (!list || index < 0) return nullptr;
    const int size = *(int*)((uint8_t*)list + OFF_LIST_SIZE);
    if (size <= index || size > 128) return nullptr;
    void* arr = *(void**)((uint8_t*)list + OFF_LIST_ITEMS);
    if (!arr) return nullptr;
    return *(void**)((uint8_t*)arr + OFF_ARRAY_ITEMS + sizeof(void*) * index);
}

static int listSize(void* list) {
    if (!list) return 0;
    const int size = *(int*)((uint8_t*)list + OFF_LIST_SIZE);
    return (size >= 0 && size <= 128) ? size : 0;
}

static int64_t localSequenceNumber(void* localGs) {
    return localGs
        ? *(const int64_t*)((const uint8_t*)localGs + OFF_SOL_CURRENT_SEQ)
        : -1;
}

static void* localPendingMoves(void* localGs) {
    return localGs
        ? *(void**)((uint8_t*)localGs + OFF_SOL_PENDING_MOVES)
        : nullptr;
}

static bool pendingMovesContainSequence(void* localGs, int64_t sequence) {
    if (sequence < 0) return false;
    void* pending = localPendingMoves(localGs);
    for (int i = 0, count = listSize(pending); i < count; ++i) {
        void* move = listItem(pending, i);
        if (move && *(const int64_t*)((const uint8_t*)move + OFF_SOL_MOVE_SEQ) == sequence)
            return true;
    }
    return false;
}

static bool submissionQueueHasRoom(void* localGs) {
    // Preserve the game's own optimistic queue, but never let speed build an
    // unbounded rollback chain if backend reconciliation is briefly late.
    return listSize(localPendingMoves(localGs)) < SOL_OPTIMISTIC_PIPELINE_MAX;
}

static void resetLastSubmissionObservation() {
    g_lastSubmittedSequence = -1;
    g_lastSubmittedSequenceBefore = -1;
    g_lastSubmittedPendingCount = 0;
    g_lastSubmittedWasQueued = false;
    g_lastSubmittedFoundationRescueCardId = -1;
}

static void captureLastSubmissionObservation(void* localGs,
                                             int64_t sequenceBefore) {
    const int64_t sequenceAfter = localSequenceNumber(localGs);
    g_lastSubmittedSequenceBefore = sequenceBefore;
    g_lastSubmittedPendingCount = listSize(localPendingMoves(localGs));
    // LogMove increments CurrentseqNumber synchronously in optimistic mode.
    // In server-authoritative mode no local sequence may be created; -1 keeps
    // that distinction explicit instead of pretending the dispatch was ACKed.
    g_lastSubmittedSequence = sequenceAfter > sequenceBefore
        ? sequenceAfter
        : -1;
    g_lastSubmittedWasQueued =
        pendingMovesContainSequence(localGs, g_lastSubmittedSequence);
}

static void* arrayValue(void* arrayToken, int index) {
    if (!arrayToken || index < 0) return nullptr;
    void* values = *(void**)((uint8_t*)arrayToken + OFF_JARRAY_VALUES);
    return listItem(values, index);
}

static int tokenInt(void* token) {
    if (!token) return -1;
    void* boxed = *(void**)((uint8_t*)token + OFF_JVALUE_VALUE);
    if (!boxed) return -1;
    void* klass = *(void**)boxed;
    const char* className = klass && g_classGetName ? g_classGetName(klass) : nullptr;
    if (className && std::string(className) == "String") {
        // GameStore's Newtonsoft tree stores Solitaire mapping values as JSON
        // strings (for example "24" and "-1"), not boxed Int32 values.
        // Reading the first eight UTF-16 bytes as an int64 made every raw
        // GameStore mapping look empty even though the property was present.
        const std::string text = readIl2CppString(boxed);
        if (text.empty()) return -1;
        size_t index = 0;
        bool negative = false;
        if (text[index] == '-') {
            negative = true;
            if (++index == text.size()) return -1;
        }
        int value = 0;
        for (; index < text.size(); ++index) {
            const char digit = text[index];
            if (digit < '0' || digit > '9') return -1;
            value = value * 10 + (digit - '0');
            if (value > 1000000) return -1;
        }
        return negative ? -value : value;
    }
    if (className && std::string(className) == "Int32") {
        const int32_t value = *(int32_t*)((uint8_t*)boxed + 0x10);
        return (value >= 0 && value <= 1000000) ? value : -1;
    }
    if (className && std::string(className) == "UInt32") {
        const uint32_t value = *(uint32_t*)((uint8_t*)boxed + 0x10);
        return value <= 1000000 ? (int)value : -1;
    }
    const int64_t value = *(int64_t*)((uint8_t*)boxed + 0x10);
    return (value >= 0 && value <= 1000000) ? (int)value : -1;
}

static std::string tokenString(void* token) {
    if (!token) return {};
    void* value = *(void**)((uint8_t*)token + OFF_JVALUE_VALUE);
    return readIl2CppString(value);
}

// HsStacks is the game's authoritative hidden-state JObject (SolitaireLocalGs
// +0x118 in V28.3.16). It is intentionally diagnostic-only here: dumping it
// once per round lets us verify whether this offline mode exposes additional
// card metadata without making the solver depend on an unverified JSON shape.
static void logHiddenStateSnapshot(void* localGs) {
    if (!localGs || g_loggedHiddenStateSnapshot || !g_objectToString) return;
    void* hiddenState = *(void**)((uint8_t*)localGs + OFF_SOL_HS_STACKS);
    if (!hiddenState) return;
    void* exception = nullptr;
    void* serialized = g_objectToString(hiddenState, &exception);
    if (exception || !serialized) {
        SOL_LOG("hs-stacks ptr=%p serialization-failed exception=%p",
                hiddenState, exception);
        g_loggedHiddenStateSnapshot = true;
        return;
    }
    const std::string json = readIl2CppString(serialized);
    constexpr size_t kLogChunk = 2800;
    const size_t parts = std::max<size_t>(1, (json.size() + kLogChunk - 1) / kLogChunk);
    for (size_t part = 0; part < parts; ++part) {
        const size_t offset = part * kLogChunk;
        const size_t count = std::min(kLogChunk, json.size() - offset);
        SOL_LOG("hs-stacks part=%d/%d json=%.*s", (int)part + 1, (int)parts,
                (int)count, json.data() + offset);
    }
    g_loggedHiddenStateSnapshot = true;
}

static void* objectField(void* object, const char* wanted) {
    if (!object || !wanted) return nullptr;
    void* properties = *(void**)((uint8_t*)object + OFF_JOBJECT_PROPERTIES);
    void* items = properties ? *(void**)((uint8_t*)properties + OFF_COLLECTION_ITEMS) : nullptr;
    const int count = listSize(items);
    for (int i = 0; i < count; ++i) {
        void* property = listItem(items, i);
        if (!property) continue;
        const std::string name = readIl2CppString(*(void**)((uint8_t*)property + OFF_JPROPERTY_NAME));
        if (name != wanted) continue;
        void* content = *(void**)((uint8_t*)property + OFF_JPROPERTY_CONTENT);
        return content ? *(void**)((uint8_t*)content + OFF_JPROPERTYLIST_TOKEN) : nullptr;
    }
    return nullptr;
}

static void* objectFieldAny(void* object, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (void* value = objectField(object, name)) return value;
    }
    return nullptr;
}

static Card parseCard(void* token, bool includeColor = false) {
    Card card;
    // The current Solitaire state stores each card as a JObject.  Keep the
    // JArray route for backwards compatibility with older local-state builds.
    void* id = objectFieldAny(token, {"id", "Id", "ID", "cardId", "CardId"});
    void* value = objectFieldAny(token, {"val", "value", "Value", "cardValue", "CardValue"});
    card.id = tokenInt(id);
    card.value = tokenInt(value);
    if (card.value >= 0 && card.value < 52) card.knownValue = card.value;
    // The solver derives suit/color from the value. Reading the serialized
    // color for every card on every tick only allocates strings; diagnostics
    // can explicitly request it when dumping a raw snapshot.
    if (includeColor)
        card.color = tokenString(objectFieldAny(token, {"color", "Color", "suit", "Suit"}));
    if (card.id >= 0 || card.value >= 0) return card;

    card.id = tokenInt(arrayValue(token, 0));
    card.value = tokenInt(arrayValue(token, 1));
    if (card.value >= 0 && card.value < 52) card.knownValue = card.value;

    // CardController.ToJArray serializes the color as a nested JArray.  Keep
    // the direct form as a fallback so the reader remains tolerant of a minor
    // game-side serialization change.
    if (includeColor) {
        void* colorToken = arrayValue(token, 2);
        card.color = tokenString(colorToken);
        if (card.color.empty()) card.color = tokenString(arrayValue(colorToken, 0));
    }
    return card;
}

static bool isRevealed(void* localGs, int cardId) {
    if (!localGs || cardId < 0) return false;
    void* set = *(void**)((uint8_t*)localGs + OFF_SOL_REVEALED_IDS);
    if (!set) return false;
    const int lastIndex = *(int*)((uint8_t*)set + OFF_HASHSET_LAST_INDEX);
    if (lastIndex < 0 || lastIndex > 80) return false;
    void* slots = *(void**)((uint8_t*)set + OFF_HASHSET_SLOTS);
    if (!slots) return false;

    // HashSet<int>.Slot is { hashCode, next, value }.  Reading the real
    // RevealedCardIds set lets us move an entire exposed sequence rather than
    // only the last card of a tableau column.
    const uint8_t* values = (const uint8_t*)slots + OFF_ARRAY_ITEMS;
    for (int i = 0; i < lastIndex; ++i) {
        const int* slot = (const int*)(values + i * 12);
        if (slot[0] >= 0 && slot[2] == cardId) return true;
    }
    return false;
}

static Card firstMovableTableauCard(void* localGs, const std::vector<Card>& cards) {
    (void)localGs;
    // In this local state, a closed card is encoded as val=-1.  The first
    // non-negative value is precisely the start of the exposed sequence.
    for (const Card& card : cards) {
        if (card.faceUp()) return card;
    }
    return {};
}

// Verified against SolitaireRaycaster.Rank/Suit/CanDropToTableauStack and
// CanDropToFoundationStack in the shipped IL2CPP binary.
static int cardRank(const Card& card) {
    if (!card.faceUp()) return -1;
    return (card.value % 13 + 1) % 13; // Ace=0 ... King=12
}

static int cardSuit(const Card& card) {
    return card.faceUp() ? (card.value / 13) : -1;
}

static int knownCardRank(const Card& card) {
    const int value = card.faceUp() ? card.value : card.knownValue;
    return (value >= 0 && value < 52) ? (value % 13 + 1) % 13 : -1;
}

static int knownCardSuit(const Card& card) {
    const int value = card.faceUp() ? card.value : card.knownValue;
    return (value >= 0 && value < 52) ? value / 13 : -1;
}

static bool canDropOnTableau(const Card& moving, const Card* destination) {
    const int movingRank = cardRank(moving);
    if (movingRank < 0) return false;
    if (!destination) return movingRank == 12; // only a King opens a column
    const int destRank = cardRank(*destination);
    if (destRank < 0) return false;
    return destRank - movingRank == 1 && ((cardSuit(*destination) & 1) != (cardSuit(moving) & 1));
}

static bool canDropOnFoundation(const Card& moving, const Card* destination) {
    const int movingRank = cardRank(moving);
    if (movingRank < 0) return false;
    if (!destination) return movingRank == 0; // Ace starts a foundation
    const int destRank = cardRank(*destination);
    return destRank >= 0 && cardSuit(moving) == cardSuit(*destination) && movingRank == destRank + 1;
}

static std::array<int, 4> foundationRanksBySuit(
        const std::array<std::vector<Card>, 4>& foundations) {
    std::array<int, 4> ranks{-1, -1, -1, -1};
    for (const auto& pile : foundations) {
        if (pile.empty()) continue;
        const int suit = cardSuit(pile.back()), rank = cardRank(pile.back());
        if (suit >= 0 && suit < 4 && rank >= 0 && rank < 13)
            ranks[suit] = std::max(ranks[suit], rank);
    }
    return ranks;
}

// Published red/black Tableau dominance with Foundation worry-back. This
// classifies strategic safety, NOT legality. Draw-3 Waste moves additionally
// need a searched continuation: removing one can change stock accessibility.
static int safeFoundationRank(int suit, const std::array<int, 4>& ranks) {
    const int otherSameColor = suit ^ 2; // native suit color is suit & 1
    const int opposite = std::min(ranks[suit ^ 1], ranks[suit ^ 3]);
    return std::max(1, std::min(opposite + 2, ranks[otherSameColor] + 3));
}

static bool foundationAdvanceIsSafe(const Card& moving,
                                    const std::array<std::vector<Card>, 4>& foundations) {
    const int rank = cardRank(moving); // Ace=0, Two=1, ... King=12
    const int suit = cardSuit(moving);
    if (rank < 0 || suit < 0) return false;
    if (rank <= 1) return true; // Aces and twos are always free progress.

    // Foundation slots are dynamic, so derive progress by suit.  A missing
    // opposite-colour suit is rank -1, not "unknown": treating it as absent
    // from the minimum was allowing one advanced black/red pile to make an
    // unsafe push look harmless while the other suit was still at zero.
    return suit < 4 && rank <= safeFoundationRank(suit, foundationRanksBySuit(foundations));
}

// Measures how many ranks the four foundations have advanced beyond the
// conservative opposite- and other-same-colour thresholds. Zero means every exposed
// foundation card is still available at a tactically safe height.  Keeping
// this as a debt (instead of a binary flag) lets deep search temporarily push
// one risky blocker only when the same calculated line repays that risk by
// uncovering a card or advancing the lagging foundations.
static int foundationRiskDebt(const std::array<std::vector<Card>, 4>& foundations) {
    const auto topBySuit = foundationRanksBySuit(foundations);

    int debt = 0;
    for (int suit = 0; suit < 4; ++suit) {
        const int rank = topBySuit[suit];
        if (rank <= 1) continue; // Aces and twos never create tactical debt.
        debt += std::max(0, rank - safeFoundationRank(suit, topBySuit));
    }
    return debt;
}

static int firstFaceUpIndex(const std::vector<Card>& cards) {
    for (int i = 0; i < (int)cards.size(); ++i) {
        if (cards[i].faceUp()) return i;
    }
    return -1;
}

static int movableCardIndex(const std::vector<Card>& cards, int movingId) {
    const int first = firstFaceUpIndex(cards);
    for (int i = first; i >= 0 && i < (int)cards.size(); ++i) {
        if (cards[i].id == movingId) return i;
    }
    return -1;
}

static std::string tableauPositionKey(const std::array<std::vector<Card>, 7>& tableau) {
    std::string key;
    key.reserve(220);
    for (const auto& column : tableau) {
        for (const Card& card : column) {
            key += std::to_string(card.id);
            key.push_back(',');
        }
        key.push_back('|');
    }
    return key;
}

static void rememberTableauPosition(const std::array<std::vector<Card>, 7>& tableau, uint64_t now) {
    const std::string key = tableauPositionKey(tableau);
    if (!g_positionHistory.empty() && g_positionHistory.back().tableau == key) return;
    g_positionHistory.push_back({key, now});
    while (g_positionHistory.size() > 40 ||
           (!g_positionHistory.empty() && now - g_positionHistory.front().at > 90000)) {
        g_positionHistory.erase(g_positionHistory.begin());
    }
}

static bool wouldRepeatRecentTableau(const std::array<std::vector<Card>, 7>& tableau,
                                     int from, int to, int movingId, uint64_t now) {
    if (from < 0 || from >= 7 || to < 0 || to >= 7 || from == to) return true;
    const int start = movableCardIndex(tableau[from], movingId);
    if (start < 0) return true;
    auto simulated = tableau;
    std::vector<Card> sequence(simulated[from].begin() + start, simulated[from].end());
    simulated[from].erase(simulated[from].begin() + start, simulated[from].end());
    simulated[to].insert(simulated[to].end(), sequence.begin(), sequence.end());
    const std::string key = tableauPositionKey(simulated);
    for (auto it = g_positionHistory.rbegin(); it != g_positionHistory.rend(); ++it) {
        if (now - it->at > 120000) break;
        if (it->tableau == key) return true;
    }
    return false;
}

static int countProductiveOptions(const std::array<std::vector<Card>, 7>& tableau,
                                  const std::array<std::vector<Card>, 4>& foundations) {
    int options = 0;
    // Count moves that expose a closed card.
    for (int src = 0; src < 7; ++src) {
        const int start = firstFaceUpIndex(tableau[src]);
        if (start < 0 || start == 0) continue;
        const Card& moving = tableau[src][start];
        for (int dst = 0; dst < 7; ++dst) {
            if (src == dst) continue;
            const Card* top = tableau[dst].empty() ? nullptr : &tableau[dst].back();
            if (canDropOnTableau(moving, top)) ++options;
        }
    }
    // Count safe advances to foundation, which also reduce congestion.
    for (int src = 0; src < 7; ++src) {
        if (tableau[src].empty() || !tableau[src].back().faceUp()) continue;
        const Card& moving = tableau[src].back();
        if (!foundationAdvanceIsSafe(moving, foundations)) continue;
        for (int f = 0; f < 4; ++f) {
            const Card* top = foundations[f].empty() ? nullptr : &foundations[f].back();
            if (canDropOnFoundation(moving, top)) { ++options; break; }
        }
    }
    return options;
}

static int structuralScore(const std::array<std::vector<Card>, 7>& tableau,
                           const std::array<std::vector<Card>, 4>& foundations) {
    int hidden = 0;
    int orderedLinks = 0;
    int foundationCards = 0;
    bool availableKing = false;
    for (const auto& column : tableau) {
        for (int i = 0; i < (int)column.size(); ++i) {
            if (!column[i].faceUp()) { ++hidden; continue; }
            if (cardRank(column[i]) == 12) availableKing = true;
            if (i > 0 && column[i - 1].faceUp() &&
                canDropOnTableau(column[i], &column[i - 1])) ++orderedLinks;
        }
    }
    int usefulEmptyColumns = 0;
    for (const auto& column : tableau) {
        if (column.empty() && availableKing) ++usefulEmptyColumns;
    }
    for (const auto& pile : foundations) foundationCards += (int)pile.size();
    return foundationCards * 24 + orderedLinks * 14 + usefulEmptyColumns * 18 - hidden * 120;
}

static bool improvesNextPosition(const std::array<std::vector<Card>, 7>& tableau,
                                 const std::array<std::vector<Card>, 4>& foundations,
                                 int from, int to, int movingId) {
    if (from < 0 || from >= 7 || to < 0 || to >= 7 || from == to) return false;
    const int start = movableCardIndex(tableau[from], movingId);
    if (start < 0) return false;
    const int before = countProductiveOptions(tableau, foundations);
    auto simulated = tableau;
    std::vector<Card> sequence(simulated[from].begin() + start, simulated[from].end());
    simulated[from].erase(simulated[from].begin() + start, simulated[from].end());
    simulated[to].insert(simulated[to].end(), sequence.begin(), sequence.end());
    const int after = countProductiveOptions(simulated, foundations);
    // Equal option counts are useful only when this exact move exposes the
    // face-down card below the moved sequence. This preserves productive
    // preparation while rejecting cosmetic tableau shuffles.
    const bool revealedSomething = start > 0 && !tableau[from][start - 1].faceUp();
    return after > before || (after == before && revealedSomething);
}

static bool hasImmediatelyMovableKing(
        const std::array<std::vector<Card>, 7>& tableau, int excludedSource) {
    for (int src = 0; src < 7; ++src) {
        if (src == excludedSource) continue;
        const int first = firstFaceUpIndex(tableau[src]);
        if (first < 0) continue;
        for (int start = first; start < (int)tableau[src].size(); ++start) {
            // A King at index zero already owns the base of a column. Moving it
            // to a newly emptied column is structural churn, not new mobility.
            if (start > 0 && cardRank(tableau[src][start]) == 12) return true;
        }
    }
    return false;
}

static bool isStrategicTableauMove(const std::array<std::vector<Card>, 7>& tableau,
                                   int from, int start, int to, const Card& moving) {
    if (from < 0 || from >= 7 || to < 0 || to >= 7 || from == to) return false;
    if (start < 0 || start >= (int)tableau[from].size() || !moving.faceUp()) return false;
    const Card* top = tableau[to].empty() ? nullptr : &tableau[to].back();
    if (!canDropOnTableau(moving, top)) return false;
    const bool destinationEmpty = tableau[to].empty();
    const bool sourceWillBeEmpty = (start == 0);
    if (destinationEmpty && cardRank(moving) != 12) return false;
    // A King starting at index zero is already the base of its source stack.
    // Moving it to an empty column changes no tactical relationship and is
    // the classic King-to-empty-column loop. Only a sub-sequence (start > 0)
    // may use an empty column.
    if (destinationEmpty && cardRank(moving) == 12 && start == 0) return false;
    // Whether a non-King clear has a useful future King depends on Waste/Talon
    // state, which is unavailable in this low-level legality helper. The full
    // GameState generator applies that information-aware restriction below.
    (void)sourceWillBeEmpty;
    return true;
}

// A compact, entirely native snapshot used only for planning.  It deliberately
// contains no IL2CPP pointers: searching this state cannot mutate Unity or race
// the animation system.
struct GameState {
    std::array<std::vector<Card>, 7> tableau;
    std::array<std::vector<Card>, 4> foundations;
    std::vector<Card> talon;
    std::vector<Card> waste;
    int recycles = 0;
    int stagnantRecycles = 0;
    int drawCount = 3;
};

enum class VirtualMoveKind : uint8_t { Tableau, Foundation, FoundationToTableau, Talon };
struct Move {
    VirtualMoveKind kind = VirtualMoveKind::Tableau;
    int from = -1;       // tableau source; -1 means waste
    int to = -1;         // tableau/foundation destination
    int start = -1;      // first card of a tableau sequence
};

static int virtualFoundationCount(const GameState& state) {
    int count = 0;
    for (const auto& pile : state.foundations) count += (int)pile.size();
    return count;
}

static bool hasOpaqueCards(const GameState& state) {
    for (const auto& column : state.tableau) {
        for (const Card& card : column) if (!card.faceUp() && !card.known()) return true;
    }
    return false;
}

static int knownCardCount(const GameState& state) {
    int known = 0;
    const auto countPile = [&known](const std::vector<Card>& pile) {
        for (const Card& card : pile) if (card.known()) ++known;
    };
    for (const auto& pile : state.tableau) countPile(pile);
    for (const auto& pile : state.foundations) countPile(pile);
    countPile(state.talon);
    countPile(state.waste);
    return known;
}

static bool completeKnownDeck(const GameState& state) {
    std::array<bool, 52> ids{};
    std::array<bool, 52> values{};
    int count = 0;
    const auto checkPile = [&](const std::vector<Card>& pile) {
        for (const Card& card : pile) {
            const int value = card.faceUp() ? card.value : card.knownValue;
            if (card.id < 0 || card.id >= 52 || value < 0 || value >= 52 ||
                ids[(size_t)card.id] || values[(size_t)value]) return false;
            ids[(size_t)card.id] = true;
            values[(size_t)value] = true;
            ++count;
        }
        return true;
    };
    for (const auto& pile : state.tableau) if (!checkPile(pile)) return false;
    for (const auto& pile : state.foundations) if (!checkPile(pile)) return false;
    return checkPile(state.talon) && checkPile(state.waste) && count == 52;
}

static bool hasReachableKingForNewColumn(const GameState& state,
                                         int excludedSource,
                                         bool allowDistantKing = false) {
    if (hasImmediatelyMovableKing(state.tableau, excludedSource)) return true;
    if (!state.waste.empty() && knownCardRank(state.waste.back()) == 12) return true;

    // Normal play counts only the next two Draw-N exposures before giving up
    // a column. Exhausted perfect-information recovery may inspect every
    // reachable exposure: the solver can then prove a longer continuation
    // rather than permanently pruning a legal preparatory clear.
    const int draw = std::clamp(state.drawCount, 1, 3);
    for (int offset = 0, batch = 0;
         offset < (int)state.talon.size() && (allowDistantKing || batch < 2);
         offset += draw, ++batch) {
        const int exposed = std::min(offset + draw - 1,
                                     (int)state.talon.size() - 1);
        if (knownCardRank(state.talon[exposed]) == 12) return true;
    }
    return false;
}

static void revealVirtualTop(std::vector<Card>& pile) {
    if (pile.empty() || pile.back().faceUp() || pile.back().virtuallyRevealed) return;
    if (pile.back().known()) {
        pile.back().value = pile.back().knownValue;
    } else {
        // The physical move will reveal the real value on the following tick.
        // Previously this returned without changing the virtual state, so a
        // perfectly legal line that exposed an unresolved card could never
        // satisfy hiddenGain and was pruned as a dead end.
        pile.back().virtuallyRevealed = true;
    }
}

static std::string virtualPositionKey(const GameState& state) {
    std::string key;
    key.reserve(520);
    const auto append = [&key](const std::vector<Card>& pile) {
        for (const Card& card : pile) {
            key += std::to_string(card.id);
            key.push_back(card.faceUp() ? 'u' :
                          (card.virtuallyRevealed ? 'r' : 'd'));
            // A position with an unresolved face-down card is not equivalent
            // to the same physical layout after PlayerCardsMapping has
            // supplied its value.  Omitting this field allowed an opaque
            // transposition/dead-state result to poison a later perfect-info
            // search for the same card IDs.
            key.push_back(':');
            key += std::to_string(card.faceUp() ? card.value : card.knownValue);
            key.push_back(',');
        }
        key.push_back('|');
    };
    for (const auto& pile : state.tableau) append(pile);
    for (const auto& pile : state.foundations) append(pile);
    append(state.talon);
    append(state.waste);
    // Recycle count is a score penalty, not a distinct card position.  Leaving
    // it in the key made an identical Draw-N loop look new forever.  Exact
    // talon/waste order already describes reachability completely.
    key += "d" + std::to_string(state.drawCount);
    return key;
}

// The planner key above includes knowledge learned from PlayerCardsMapping.
// That knowledge can advance while Unity is still animating an unchanged
// board. A pending action must instead be acknowledged by physical card IDs,
// pile order and visibility only. A virtual reveal of an opaque card is an
// expected face-up card even though its value is learned on the next snapshot.
static std::string physicalPositionKey(const GameState& state) {
    std::string key;
    key.reserve(280);
    const auto append = [&key](const std::vector<Card>& pile,
                               bool trackVisibility) {
        for (const Card& card : pile) {
            key += std::to_string(card.id);
            // Stock/Waste can reveal their values as a side effect of a draw;
            // their changed pile membership already proves that action. Only
            // Tableau visibility is an independent physical reveal signal.
            if (trackVisibility)
                key.push_back(card.revealedForPlanning() ? 'u' : 'd');
            key.push_back(',');
        }
        key.push_back('|');
    };
    for (const auto& pile : state.tableau) append(pile, true);
    for (const auto& pile : state.foundations) append(pile, false);
    append(state.talon, false);
    append(state.waste, false);
    key += "d" + std::to_string(state.drawCount);
    return key;
}

static void appendSolverCard(std::string& key, const Card& card,
                             bool includeVisibility) {
    const int value = card.faceUp() ? card.value : card.knownValue;
    const int visibility = includeVisibility ?
        (card.faceUp() ? 1 : (card.virtuallyRevealed ? 2 : 0)) : 0;
    if (card.id >= 0 && card.id < 52 && value >= -1 && value < 52) {
        // 6 id bits, 6 value bits (zero means unknown), 2 visibility bits.
        // The top two bits remain clear, so pile separators cannot collide.
        const unsigned packed = (unsigned)card.id |
            ((unsigned)(value + 1) << 6) | ((unsigned)visibility << 12);
        key.push_back((char)(packed & 0xff));
        key.push_back((char)(packed >> 8));
        return;
    }
    // Diagnostic/synthetic states can have IDs outside a real 52-card deal.
    // Keep their transposition keys exact instead of truncating them.
    key.push_back((char)0xff);
    key.push_back((char)0xfc);
    const auto appendInt = [&key](int number) {
        const uint32_t bits = (uint32_t)number;
        for (int shift = 0; shift < 32; shift += 8)
            key.push_back((char)(bits >> shift));
    };
    appendInt(card.id);
    appendInt(value);
    key.push_back((char)visibility);
}

static std::string solverPositionKey(const GameState& state) {
    // Tableau columns are rule-symmetric. Canonicalising their encodings makes
    // permutations of the same board share one transposition entry, yielding a
    // much deeper search for the same Unity-thread time budget. Physical move
    // validation continues to use virtualPositionKey with real column indices.
    std::array<std::string, 7> columns;
    for (int i = 0; i < 7; ++i) {
        std::string& encoded = columns[i];
        encoded.reserve(state.tableau[i].size() * 2);
        for (const Card& card : state.tableau[i])
            appendSolverCard(encoded, card, true);
    }
    std::sort(columns.begin(), columns.end());

    std::string key;
    key.reserve(144);
    for (const std::string& column : columns) {
        key += column;
        key.push_back((char)0xff);
        key.push_back((char)0xff);
    }
    std::array<int, 4> foundationBySuit{-1, -1, -1, -1};
    for (const auto& pile : state.foundations) {
        if (pile.empty()) continue;
        const int suit = knownCardSuit(pile.back());
        if (suit >= 0 && suit < 4) foundationBySuit[suit] = knownCardRank(pile.back());
    }
    key.push_back((char)0xff);
    key.push_back((char)0xfe);
    for (int rank : foundationBySuit) key.push_back((char)(rank + 1));
    const auto appendOrdered = [&key](const std::vector<Card>& pile) {
        key.push_back((char)0xff);
        key.push_back((char)0xfd);
        for (const Card& card : pile) appendSolverCard(key, card, false);
    };
    appendOrdered(state.talon);
    appendOrdered(state.waste);
    key += "|d" + std::to_string(state.drawCount);
    return key;
}

// Exact recent-history identity (including physical pile indices), encoded
// with the existing collision-free card representation. The printable key
// remains available for dispatch diagnostics; search no longer formats 104
// decimal integers per child merely to check recent-board membership.
static std::string recentPositionKey(const GameState& state) {
    std::string key;
    key.reserve(144);
    const auto append = [&key](const std::vector<Card>& pile) {
        for (const Card& card : pile) appendSolverCard(key, card, true);
        key.push_back((char)0xff);
        key.push_back((char)0xff);
    };
    for (const auto& pile : state.tableau) append(pile);
    for (const auto& pile : state.foundations) append(pile);
    append(state.talon);
    append(state.waste);
    key += 'd';
    key += std::to_string(state.drawCount);
    return key;
}

static int virtualMobility(const GameState& state) {
    int moves = 0;
    const auto countTableauTargets = [&](const Card& card) {
        for (int dst = 0; dst < 7; ++dst) {
            const Card* top = state.tableau[dst].empty() ? nullptr : &state.tableau[dst].back();
            if (canDropOnTableau(card, top)) ++moves;
        }
    };
    if (!state.waste.empty() && state.waste.back().faceUp()) countTableauTargets(state.waste.back());
    for (int src = 0; src < 7; ++src) {
        const int first = firstFaceUpIndex(state.tableau[src]);
        for (int start = first; start >= 0 && start < (int)state.tableau[src].size(); ++start) {
            const Card& card = state.tableau[src][start];
            if (!card.faceUp()) continue;
            for (int dst = 0; dst < 7; ++dst) {
                if (dst == src) continue;
                const Card* top = state.tableau[dst].empty() ? nullptr : &state.tableau[dst].back();
                if (canDropOnTableau(card, top)) ++moves;
            }
        }
    }
    return moves;
}

static int dependencyPenalty(const GameState& state) {
    std::array<int, 4> neededRank{0, 0, 0, 0};
    for (const auto& pile : state.foundations) {
        if (pile.empty()) continue;
        const int suit = knownCardSuit(pile.back());
        const int rank = knownCardRank(pile.back());
        if (suit >= 0 && suit < 4 && rank >= 0) neededRank[suit] = std::min(13, rank + 1);
    }

    int penalty = 0;
    for (int suit = 0; suit < 4; ++suit) {
        if (neededRank[suit] >= 13) continue;
        bool located = false;
        for (const auto& column : state.tableau) {
            for (int i = 0; i < (int)column.size(); ++i) {
                if (knownCardSuit(column[i]) != suit || knownCardRank(column[i]) != neededRank[suit]) continue;
                const int blockers = (int)column.size() - 1 - i;
                penalty += blockers * (neededRank[suit] <= 1 ? 240 : 100);
                if (!column[i].faceUp()) penalty += 220;
                located = true;
                break;
            }
            if (located) break;
        }
        if (located) continue;
        for (int i = (int)state.waste.size() - 1, distance = 0; i >= 0; --i, ++distance) {
            if (knownCardSuit(state.waste[i]) == suit && knownCardRank(state.waste[i]) == neededRank[suit]) {
                penalty += distance * 55;
                located = true;
                break;
            }
        }
        if (located) continue;
        // LocalGs stores the next stock card at index zero (confirmed against
        // consecutive real Draw-3 snapshots), while Waste stores its exposed
        // card at the back.  Distances must follow that asymmetric ordering.
        for (int i = 0, distance = 0; i < (int)state.talon.size(); ++i, ++distance) {
            if (knownCardSuit(state.talon[i]) == suit && knownCardRank(state.talon[i]) == neededRank[suit]) {
                const int draw = std::clamp(state.drawCount, 1, 3);
                penalty += (distance / draw) * 70 + (distance % draw) * 20;
                break;
            }
        }
    }
    return penalty;
}

// Late in a partial-information round the exact set of unknown values is much
// smaller than the set of unknown physical cards: every known value removes
// one member of the standard 52-card deck.  Do not guess which hidden card is
// which. Instead, pessimistically pair the most strategically important
// remaining values with the hardest-to-expose unknown positions. This gives
// all possible assignments a cheap lower-bound cost and steers search toward
// routes that improve the worst case. It directly covers the captured 48/52
// stalls without allocating or multiplying the search tree by 4! scenarios.
static int unresolvedValueDependencyPenalty(const GameState& state) {
    std::array<uint8_t, 52> knownValues{};
    std::array<int, 52> positionCosts{};
    int physicalCards = 0;
    int unknownCount = 0;
    bool invalidDeck = false;

    const auto observeKnown = [&](const Card& card) {
        ++physicalCards;
        const int value = card.faceUp() ? card.value : card.knownValue;
        if (value < 0 || value >= 52) return false;
        if (knownValues[value]) invalidDeck = true;
        knownValues[value] = 1;
        return true;
    };
    const auto addUnknown = [&](int cost) {
        if (unknownCount < (int)positionCosts.size())
            positionCosts[unknownCount++] = std::max(0, cost);
    };

    for (const auto& column : state.tableau) {
        for (int i = 0; i < (int)column.size(); ++i) {
            const Card& card = column[i];
            if (observeKnown(card)) continue;
            // A virtually revealed top is immediately resolved by the next
            // physical snapshot. Do not treat it as a still-buried hazard.
            if (card.virtuallyRevealed) addUnknown(0);
            else {
                const int blockers = (int)column.size() - 1 - i;
                addUnknown(24 + blockers * 18);
            }
        }
    }
    for (const auto& pile : state.foundations) {
        for (const Card& card : pile) {
            // An unknown Foundation card means the snapshot is inconsistent;
            // decline the estimate rather than assigning invented semantics.
            if (!observeKnown(card)) invalidDeck = true;
        }
    }
    const int draw = std::clamp(state.drawCount, 1, 3);
    for (int i = 0; i < (int)state.talon.size(); ++i) {
        const Card& card = state.talon[i];
        if (observeKnown(card)) continue;
        addUnknown(8 + (i / draw) * 5 + (i % draw) * 2);
    }
    for (int i = 0; i < (int)state.waste.size(); ++i) {
        const Card& card = state.waste[i];
        if (observeKnown(card)) continue;
        addUnknown(6 + ((int)state.waste.size() - 1 - i) * 5);
    }

    // This bound is meaningful only for one complete, internally consistent
    // deck. Limit it to the small late-game uncertainty window where it is
    // both strongest and essentially free in the planner hot path.
    if (invalidDeck || physicalCards != 52 || unknownCount <= 0 || unknownCount > 8)
        return 0;

    std::array<int, 52> remainingValues{};
    int remainingCount = 0;
    for (int value = 0; value < 52; ++value)
        if (!knownValues[value]) remainingValues[remainingCount++] = value;
    if (remainingCount != unknownCount) return 0;

    std::array<int, 4> neededRank{0, 0, 0, 0};
    for (const auto& pile : state.foundations) {
        if (pile.empty()) continue;
        const int suit = knownCardSuit(pile.back());
        const int rank = knownCardRank(pile.back());
        if (suit >= 0 && suit < 4 && rank >= 0)
            neededRank[suit] = std::min(13, rank + 1);
    }

    // Count demand for the opposite-colour parent cards needed to relocate
    // exposed Tableau/Waste cards. Values with several live dependants are
    // dangerous when they might occupy a deeply buried unknown position.
    std::array<int, 52> landingDemand{};
    const auto addLandingDemand = [&](const Card& card) {
        const int rank = cardRank(card);
        const int suit = cardSuit(card);
        if (rank < 0 || rank >= 12 || suit < 0) return;
        const int wantedRank = rank + 1;
        // Jawaker's value-to-rank mapping gives exactly one matching value
        // per opposite-colour suit. Avoid a 52-value scan for every exposed
        // card in every late-game search node.
        for (int parentSuit = 0; parentSuit < 4; ++parentSuit)
            if ((parentSuit & 1) != (suit & 1))
                ++landingDemand[parentSuit * 13 + wantedRank - 1];
    };
    for (const auto& column : state.tableau)
        for (const Card& card : column)
            if (card.faceUp()) addLandingDemand(card);
    if (!state.waste.empty() && state.waste.back().faceUp())
        addLandingDemand(state.waste.back());

    std::array<int, 52> valueWeights{};
    for (int i = 0; i < remainingCount; ++i) {
        const int value = remainingValues[i];
        const int suit = value / 13;
        const int rank = (value % 13 + 1) % 13;
        int weight = 1;
        if (rank == neededRank[suit]) weight += 14;
        else if (rank == neededRank[suit] + 1) weight += 5;
        if (rank <= 1) weight += 7;   // Ace/Two unlock Foundation progress.
        if (rank == 12) weight += 6;  // King is the only empty-column key.
        weight += std::min(3, landingDemand[value]) * 4;
        valueWeights[i] = weight;
    }

    std::sort(positionCosts.begin(), positionCosts.begin() + unknownCount,
              std::greater<int>());
    std::sort(valueWeights.begin(), valueWeights.begin() + remainingCount,
              std::greater<int>());
    int penalty = 0;
    for (int i = 0; i < unknownCount; ++i)
        penalty += positionCosts[i] * valueWeights[i];
    return std::min(12000, penalty);
}

// Jawaker 28.3.16's SolitaireV2AutoCompleteController checks this local layout
// before it requests auto-complete: no stock, no waste, and no concealed
// Tableau value. The separate GameStore eligibility bit is authoritative, so
// this is a search milestone, never a locally fabricated win.
static bool autoCompleteLayoutCandidate(const GameState& state) {
    if (!state.talon.empty() || !state.waste.empty()) return false;
    for (const auto& column : state.tableau)
        for (const Card& card : column)
            if (!card.revealedForPlanning()) return false;
    return true;
}

// The server's eligibility bit is not computed here. This cheap, allocation-
// free certificate only allows the planner to strongly favor a local layout
// when every remaining Tableau card can advance to Foundation without any
// additional Tableau or stock maneuver. If the server does not auto-complete,
// the same line can still be finished by ordinary legal moves.
static bool provenAutoCompleteLayout(const GameState& state,
                                     std::vector<Move>* winningLine = nullptr) {
    if (winningLine) winningLine->clear();
    if (!autoCompleteLayoutCandidate(state)) return false;
    std::array<int, 7> remaining{};
    std::array<Card, 4> foundationTop{};
    std::array<bool, 4> hasFoundationTop{};
    int foundationCount = 0;
    int tableauCount = 0;
    for (int src = 0; src < 7; ++src) {
        remaining[src] = (int)state.tableau[src].size();
        tableauCount += remaining[src];
    }
    for (int dst = 0; dst < 4; ++dst) {
        const auto& pile = state.foundations[dst];
        foundationCount += (int)pile.size();
        if (!pile.empty()) {
            foundationTop[dst] = pile.back();
            hasFoundationTop[dst] = true;
        }
    }
    if (foundationCount + tableauCount != 52) return false;
    if (foundationCount == 52) return true;
    if (winningLine) winningLine->reserve((size_t)(52 - foundationCount));
    for (int placed = foundationCount; placed < 52; ++placed) {
        bool advanced = false;
        for (int src = 0; src < 7 && !advanced; ++src) {
            if (remaining[src] == 0) continue;
            const Card& card = state.tableau[src][(size_t)remaining[src] - 1];
            if (!card.faceUp()) continue;
            for (int dst = 0; dst < 4; ++dst) {
                const Card* top = hasFoundationTop[dst] ? &foundationTop[dst] : nullptr;
                if (!canDropOnFoundation(card, top)) continue;
                foundationTop[dst] = card;
                hasFoundationTop[dst] = true;
                --remaining[src];
                if (winningLine)
                    winningLine->push_back({VirtualMoveKind::Foundation, src, dst, -1});
                advanced = true;
                break;
            }
        }
        if (!advanced) {
            if (winningLine) winningLine->clear();
            return false;
        }
    }
    for (int count : remaining) {
        if (count != 0) {
            if (winningLine) winningLine->clear();
            return false;
        }
    }
    return true;
}

static int virtualStructuralScore(const GameState& state,
                                  int autoCompleteProof = -1) {
    int hidden = 0, exposed = 0, orderedLinks = 0;
    const int foundationCards = virtualFoundationCount(state);
    int blockedHighCards = 0, buriedCritical = 0;
    bool availableKing = false;
    for (const auto& column : state.tableau) {
        const int first = firstFaceUpIndex(column);
        if (first > 0 && first < (int)column.size()) {
            // A high card immediately above hidden cards is expensive: it is
            // commonly the card that prevents the column from being emptied.
            blockedHighCards += std::max(0, cardRank(column[first]) - 5);
        }
        for (int i = 0; first > 0 && i < first; ++i) {
            const int hiddenRank = knownCardRank(column[i]);
            if (hiddenRank >= 0 && (hiddenRank <= 1 || hiddenRank == 12))
                buriedCritical += first - i;
        }
        for (int i = 0; i < (int)column.size(); ++i) {
            if (!column[i].revealedForPlanning()) { ++hidden; continue; }
            ++exposed;
            // An opaque card that was virtually exposed proves progress but
            // has no rank/suit yet.  It cannot participate in another virtual
            // move until the physical snapshot resolves it.
            if (!column[i].faceUp()) continue;
            // A King already anchoring index zero cannot productively move to
            // another empty column. Any other exposed King can consume a newly
            // opened space and unlock the cards beneath its source.
            if (cardRank(column[i]) == 12 && i > 0) availableKing = true;
            if (i > 0 && column[i - 1].faceUp() &&
                canDropOnTableau(column[i], &column[i - 1])) ++orderedLinks;
        }
    }
    // The draw end is the next exposed card. Deep Aces, twos and Kings cost
    // future passes because they are strategically critical unlockers.
    for (int i = 0; i < (int)state.talon.size(); ++i) {
        const int rank = knownCardRank(state.talon[i]);
        const int distance = i;
        if (rank >= 0 && (rank <= 1 || rank == 12)) buriedCritical += distance + 1;
    }
    if (!state.waste.empty() && cardRank(state.waste.back()) == 12) availableKing = true;

    int emptyColumns = 0;
    for (const auto& column : state.tableau) if (column.empty()) ++emptyColumns;
    // Move generation already treats a King exposed by the next two Draw-N
    // steps as usable. The evaluator used to score that *same* legal empty
    // column as congestion because it checked only the current Waste top.
    // Apply the generator's bounded rule here as well; no distant or unknown
    // stock card is invented merely to inflate an empty-column score.
    if (emptyColumns > 0 && !availableKing)
        availableKing = hasReachableKingForNewColumn(state, -1);
    // An empty column is valuable only when a useful King can consume it.  A
    // vacant column without such a King is congestion in disguise: the bot
    // paid a move and reduced its construction surface for no immediate gain.
    const int emptyColumnValue = availableKing ? emptyColumns * 850 : -emptyColumns * 700;

    // Balanced foundations are safer than one suit racing far ahead.  This
    // encodes the same tactical reason as foundationAdvanceIsSafe in the global
    // evaluator, so a deep branch can trade a small imbalance for a reveal but
    // will not drift into a permanently trapped colour.
    std::array<int, 4> foundationRank{-1, -1, -1, -1};
    for (const auto& pile : state.foundations) {
        if (pile.empty()) continue;
        const int suit = knownCardSuit(pile.back());
        if (suit >= 0 && suit < 4) foundationRank[suit] = knownCardRank(pile.back());
    }
    const int minFoundation = *std::min_element(foundationRank.begin(), foundationRank.end());
    const int maxFoundation = *std::max_element(foundationRank.begin(), foundationRank.end());
    const int imbalancePenalty = std::max(0, maxFoundation - minFoundation - 2) * 360;
    const int foundationDebtPenalty = foundationRiskDebt(state.foundations) * 1800;

    // Once the hidden cards are gone, draining stock and waste can end the
    // round before 52 physical Foundation moves. Keep this below a proved
    // 52-card win and only award it at the exact local eligibility layout.
    const int autoCompleteLayoutBonus =
        hidden == 0 && foundationCards < 52 &&
        (autoCompleteProof >= 0 ? autoCompleteProof != 0 :
         provenAutoCompleteLayout(state)) ? 60000 : 0;
    return autoCompleteLayoutBonus + foundationCards * 260 + exposed * 32 + orderedLinks * 75 +
           virtualMobility(state) * 70 + emptyColumnValue - hidden * 2200 -
           blockedHighCards * 130 - buriedCritical * 14 - imbalancePenalty -
           foundationDebtPenalty - dependencyPenalty(state) -
           unresolvedValueDependencyPenalty(state);
}

static bool applyVirtualMove(GameState& state, const Move& move) {
    if (move.kind == VirtualMoveKind::Talon) {
        if (!state.talon.empty()) {
            // Jawaker's Solo mode presents a three-card draw; preserving order
            // here lets planning evaluate future waste exposure accurately.
            const int draw = std::min(std::clamp(state.drawCount, 1, 3),
                                      (int)state.talon.size());
            for (int i = 0; i < draw; ++i) {
                Card drawn = state.talon[(size_t)i];
                if (!drawn.faceUp() && drawn.known()) drawn.value = drawn.knownValue;
                state.waste.push_back(std::move(drawn));
            }
            state.talon.erase(state.talon.begin(), state.talon.begin() + draw);
            return true;
        }
        if (state.waste.empty()) return false;
        // LocalGs keeps real values in talon/waste.  Preserve them so a full
        // recycle returns to the exact same native state key and the solver's
        // transposition table can prove the loop instead of inventing a new
        // face-down variant on every pass.
        for (Card& card : state.waste)
            if (card.faceUp()) card.knownValue = card.value;
        // Waste is oldest-to-newest and its back is exposed. Turning the pile
        // over makes the oldest remaining card the next stock card, exactly
        // the same front-to-back vector order; reversing here used to corrupt
        // every planned path after the first recycle.
        state.talon.swap(state.waste);
        ++state.recycles;
        ++state.stagnantRecycles;
        return true;
    }
    if (move.kind == VirtualMoveKind::Foundation) {
        Card card;
        if (move.from < 0) {
            if (state.waste.empty()) return false;
            card = state.waste.back();
        } else {
            if (move.from >= 7 || state.tableau[move.from].empty()) return false;
            card = state.tableau[move.from].back();
        }
        if (move.to < 0 || move.to >= 4 || !card.faceUp()) return false;
        const Card* top = state.foundations[move.to].empty() ? nullptr : &state.foundations[move.to].back();
        if (!canDropOnFoundation(card, top)) return false;
        if (move.from < 0) {
            state.waste.pop_back();
        } else {
            state.tableau[move.from].pop_back();
            revealVirtualTop(state.tableau[move.from]);
        }
        state.foundations[move.to].push_back(card);
        state.stagnantRecycles = 0;
        return true;
    }
    if (move.kind == VirtualMoveKind::FoundationToTableau) {
        if (move.from < 0 || move.from >= 4 || move.to < 0 || move.to >= 7 ||
            state.foundations[move.from].empty()) return false;
        const Card card = state.foundations[move.from].back();
        const Card* top = state.tableau[move.to].empty() ? nullptr : &state.tableau[move.to].back();
        if (!canDropOnTableau(card, top)) return false;
        state.foundations[move.from].pop_back();
        state.tableau[move.to].push_back(card);
        return true;
    }
    if (move.to < 0 || move.to >= 7) return false;
    if (move.from < 0) {
        if (state.waste.empty() || !state.waste.back().faceUp()) return false;
        const Card& card = state.waste.back();
        const Card* top = state.tableau[move.to].empty() ? nullptr : &state.tableau[move.to].back();
        if (!canDropOnTableau(card, top)) return false;
        state.tableau[move.to].push_back(card);
        state.waste.pop_back();
        state.stagnantRecycles = 0;
        return true;
    }
    if (move.from >= 7) return false;
    const int start = move.start;
    if (start < 0 || start >= (int)state.tableau[move.from].size()) return false;
    const Card& card = state.tableau[move.from][start];
    if (!isStrategicTableauMove(state.tableau, move.from, start, move.to, card)) return false;
    // Source and destination are distinct (checked by
    // isStrategicTableauMove), so copy directly into the destination before
    // truncating the source. This avoids a temporary vector on every branch.
    state.tableau[move.to].insert(state.tableau[move.to].end(),
                                  state.tableau[move.from].begin() + start,
                                  state.tableau[move.from].end());
    state.tableau[move.from].erase(state.tableau[move.from].begin() + start,
                                  state.tableau[move.from].end());
    revealVirtualTop(state.tableau[move.from]);
    state.stagnantRecycles = 0;
    return true;
}

// A returned Foundation card can provide the missing target for a whole
// exposed suffix whose removal reveals a hidden card. This is board-dependent
// evidence, not a history-dependent inverse-move ban. It is admitted in the
// normal graph; the planner still proves the continuation before dispatch.
static bool foundationReturnUnlocksReveal(const GameState& state,
                                           const Card& card, int destination) {
    for (int src = 0; src < 7; ++src) {
        if (src == destination) continue;
        const auto& pile = state.tableau[src];
        const int start = firstFaceUpIndex(pile);
        if (start > 0 && !pile[start - 1].revealedForPlanning() &&
            canDropOnTableau(pile[start], &card)) return true;
    }
    return false;
}

// Fixed bitboards index the real destinations, not an arrangement-free board.
// Preserve physical columns, stock order and all transition validation. Each
// destination is analyzed once per generation instead of once per source card.
struct MoveTargetMasks {
    std::array<std::array<uint8_t, 2>, 13> tableau{};
    std::array<std::array<uint8_t, 4>, 13> foundation{};

    explicit MoveTargetMasks(const GameState& state) {
        for (int dst = 0; dst < 7; ++dst) {
            const auto& pile = state.tableau[dst];
            const uint8_t bit = (uint8_t)(1U << dst);
            if (pile.empty()) {
                tableau[12][0] |= bit; tableau[12][1] |= bit;
            } else {
                const int rank = cardRank(pile.back()), suit = cardSuit(pile.back());
                if (rank > 0 && rank < 13 && suit >= 0 && suit < 4)
                    tableau[rank - 1][(suit & 1) ^ 1] |= bit;
            }
        }
        for (int dst = 0; dst < 4; ++dst) {
            const auto& pile = state.foundations[dst];
            const uint8_t bit = (uint8_t)(1U << dst);
            if (pile.empty()) {
                for (auto& suit : foundation[0]) suit |= bit;
            } else {
                const int rank = cardRank(pile.back()), suit = cardSuit(pile.back());
                if (rank >= 0 && rank < 12 && suit >= 0 && suit < 4)
                    foundation[rank + 1][suit] |= bit;
            }
        }
    }
    uint8_t tableauFor(const Card& card) const {
        const int rank = cardRank(card), suit = cardSuit(card);
        return rank >= 0 && rank < 13 && suit >= 0 && suit < 4
            ? tableau[rank][suit & 1] : 0;
    }
    uint8_t foundationFor(const Card& card) const {
        const int rank = cardRank(card), suit = cardSuit(card);
        return rank >= 0 && rank < 13 && suit >= 0 && suit < 4
            ? foundation[rank][suit] : 0;
    }
};

static int takeDestination(uint8_t& mask) {
    const int dst = __builtin_ctz((unsigned)mask); // caller guarantees nonzero
    mask &= (uint8_t)(mask - 1);
    return dst;
}

static void generateVirtualMoves(const GameState& state, std::vector<Move>& out,
                                 bool allowFoundationPullback = false,
                                 bool allowDistantKing = false) {
    out.clear();
    const MoveTargetMasks targets(state);
    if (!state.waste.empty() && state.waste.back().faceUp()) {
        uint8_t foundations = targets.foundationFor(state.waste.back());
        while (foundations) {
            const int f = takeDestination(foundations);
            out.push_back({VirtualMoveKind::Foundation, -1, f, -1});
        }
        uint8_t columns = targets.tableauFor(state.waste.back());
        while (columns) {
            const int dst = takeDestination(columns);
            out.push_back({VirtualMoveKind::Tableau, -1, dst, -1});
        }
    }
    for (int src = 0; src < 7; ++src) {
        if (!state.tableau[src].empty() && state.tableau[src].back().faceUp()) {
            uint8_t foundations = targets.foundationFor(state.tableau[src].back());
            while (foundations) {
                const int f = takeDestination(foundations);
                out.push_back({VirtualMoveKind::Foundation, src, f, -1});
            }
        }
        const int first = firstFaceUpIndex(state.tableau[src]);
        for (int start = first; start >= 0 && start < (int)state.tableau[src].size(); ++start) {
            const Card& card = state.tableau[src][start];
            uint8_t columns = targets.tableauFor(card) & (uint8_t)~(1U << src);
            // This stock/King test is destination-independent. Evaluate once.
            if (columns && start == 0 && cardRank(card) != 12 &&
                !hasReachableKingForNewColumn(state, src, allowDistantKing)) continue;
            while (columns) {
                const int dst = takeDestination(columns);
                if (!isStrategicTableauMove(state.tableau, src, start, dst, card)) continue;
                out.push_back({VirtualMoveKind::Tableau, src, dst, start});
            }
        }
    }
    // Admit tactical reveal bridges immediately; ordinary cosmetic returns
    // still require the widened/recovery graph and a certified continuation.
    {
        for (int src = 0; src < 4; ++src) {
            if (state.foundations[src].empty()) continue;
            const Card& card = state.foundations[src].back();
            if (!card.faceUp() || cardRank(card) < 2) continue; // never pull A/2
            uint8_t columns = targets.tableauFor(card);
            while (columns) {
                const int dst = takeDestination(columns);
                if (state.tableau[dst].empty()) continue;
                if (allowFoundationPullback || foundationReturnUnlocksReveal(state, card, dst))
                    out.push_back({VirtualMoveKind::FoundationToTableau, src, dst, -1});
            }
        }
    }
    // Stock is a normal branch, not an unconditional fallback, which permits
    // the planner to intentionally defer a currently legal waste move.
    if (!state.talon.empty() || !state.waste.empty())
        out.push_back({VirtualMoveKind::Talon, -1, -1, -1});
}

// Only a Tableau source can uncover one card. Read the predecessor in the
// existing state instead of scanning the complete board both before and after
// every virtual child. An opaque reveal is also a real information gain once
// Unity publishes its value on the next physical snapshot.
static int virtualRevealGain(const GameState& before, const Move& move,
                             bool* newlyUnknown = nullptr) {
    if (newlyUnknown) *newlyUnknown = false;
    if (move.from < 0 || move.from >= 7 ||
        (move.kind != VirtualMoveKind::Tableau &&
         move.kind != VirtualMoveKind::Foundation)) return 0;
    const auto& source = before.tableau[move.from];
    const int exposed = move.kind == VirtualMoveKind::Foundation
        ? (int)source.size() - 2 : move.start - 1;
    if (exposed < 0 || exposed >= (int)source.size() ||
        source[exposed].revealedForPlanning()) return 0;
    if (newlyUnknown) *newlyUnknown = !source[exposed].known();
    return 1;
}

struct PlannerSearchContext {
    int budget = 0;
    uint64_t deadlineMs = 0;
    std::unordered_map<std::string, int> transposition;
    std::unordered_set<std::string> path;
    bool allowFoundationPullback = false;
    bool allowDistantKing = false;
};

static bool seenRecentSolverState(const GameState& state, uint64_t now) {
    if (g_solverStateHistory.empty()) return false;
    const std::string key = recentPositionKey(state);
    for (auto it = g_solverStateHistory.rbegin(); it != g_solverStateHistory.rend(); ++it) {
        if (now - it->at > 180000) break;
        if (it->tableau == key) return true;
    }
    return false;
}

static int searchVirtualFuture(const GameState& state, int depth,
                               PlannerSearchContext& context) {
    const int base = virtualStructuralScore(state);
    if (virtualFoundationCount(state) == 52) return base + 1000000;
    if (provenAutoCompleteLayout(state)) return base + 250000;
    if (depth <= 0 || context.budget <= 0 || nowMs() >= context.deadlineMs) return base;

    const std::string stateKey = solverPositionKey(state);
    if (context.path.find(stateKey) != context.path.end()) return -8000000;
    // The same layout with one free stock pass left is not strategically
    // equivalent to one whose future recycles are all charged.
    const std::string cacheKey = stateKey + '#' + std::to_string(depth) + ':' +
        (char)('0' + std::max(0, SOL_FREE_TALON_RECYCLES - state.recycles));
    const auto cached = context.transposition.find(cacheKey);
    if (cached != context.transposition.end()) return cached->second;
    context.path.insert(stateKey);

    std::vector<Move> moves;
    generateVirtualMoves(state, moves, context.allowFoundationPullback,
                         context.allowDistantKing);
    if (moves.empty()) {
        context.path.erase(stateKey);
        return hasOpaqueCards(state) ? base - 5000 : -10000000;
    }

    struct Branch { GameState state; int heuristic; int edgeCost; };
    std::vector<Branch> branches;
    branches.reserve(moves.size());
    for (const Move& move : moves) {
        if (context.budget-- <= 0 || nowMs() >= context.deadlineMs) break;
        GameState next = state;
        if (!applyVirtualMove(next, move)) continue;
        if (virtualFoundationCount(next) == 52) {
            context.path.erase(stateKey);
            return virtualStructuralScore(next) + 1000000;
        }
        std::string key = solverPositionKey(next);
        if (context.path.find(key) != context.path.end()) continue;
        const int edgeCost = move.kind == VirtualMoveKind::Talon
            ? -SOL_TALON_STEP_SEARCH_COST - (state.talon.empty() &&
                     state.recycles >= SOL_FREE_TALON_RECYCLES
                         ? SOL_CHARGED_RECYCLE_SEARCH_COST : 0)
            : 0;
        branches.push_back({std::move(next), 0, edgeCost});
        // Ordering only: reveal branches enter the fixed-width beam before
        // cosmetic moves. This does not turn a Foundation move into a forced
        // choice or change the recursive score of the resulting board.
        branches.back().heuristic =
            virtualStructuralScore(branches.back().state) + edgeCost +
            virtualRevealGain(state, move) * 6000;
    }
    if (branches.empty()) {
        context.path.erase(stateKey);
        return hasOpaqueCards(state) ? base - 5000 : -10000000;
    }
    std::sort(branches.begin(), branches.end(), [](const Branch& a, const Branch& b) { return a.heuristic > b.heuristic; });
    int hidden = 0;
    for (const auto& column : state.tableau)
        for (const Card& card : column)
            if (!card.revealedForPlanning()) ++hidden;
    // Keep the established fast beam in the open game.  Near the end, where
    // the state space is much smaller and one temporarily unattractive move
    // can be the only winning bridge, retain more siblings instead of letting
    // the short-term heuristic kill them.  The existing node/time budget is
    // unchanged, so this cannot turn the Unity thread into an unbounded solve.
    const int requestedBeam = hidden <= 4 ? 12 : (hidden <= 9 ? 9 : 6);
    const int beamWidth = std::min(requestedBeam, (int)branches.size());
    int best = -10000000;
    for (int i = 0; i < beamWidth && context.budget > 0 && nowMs() < context.deadlineMs; ++i)
        best = std::max(best, branches[i].edgeCost +
            searchVirtualFuture(branches[i].state, depth - 1, context));
    context.path.erase(stateKey);
    context.transposition.emplace(cacheKey, best);
    return best;
}

static int candidateTacticalBonus(const GameState& before, const GameState& after,
                                   const Move& move, int knownHiddenBefore = -1,
                                   int autoCompleteProofAfter = -1) {
    int bonus = 0;
    const bool autoCompleteAfter = autoCompleteProofAfter >= 0
        ? autoCompleteProofAfter != 0 : provenAutoCompleteLayout(after);
    if (autoCompleteAfter && !autoCompleteLayoutCandidate(before)) bonus += 120000;
    bool learnsUnknown = false;
    const int revealGain = virtualRevealGain(before, move, &learnsUnknown);
    bonus += revealGain * 12000; // reveal is the primary Klondike objective.
    if (learnsUnknown) bonus += 6000; // Prefer new deal information at a fork.
    if (move.kind == VirtualMoveKind::Foundation) {
        const Card* card = nullptr;
        if (move.from < 0) {
            if (!before.waste.empty()) card = &before.waste.back();
        } else if (move.from < 7 && !before.tableau[move.from].empty()) {
            card = &before.tableau[move.from].back();
        }
        const int debtDelta = foundationRiskDebt(after.foundations) -
                              foundationRiskDebt(before.foundations);
        if (card && foundationAdvanceIsSafe(*card, before.foundations) && debtDelta <= 0) {
            int hiddenBefore = knownHiddenBefore;
            if (hiddenBefore < 0) {
                hiddenBefore = 0;
                for (const auto& pile : before.tableau)
                    for (const Card& c : pile)
                        if (!c.revealedForPlanning()) ++hiddenBefore;
            }
            bonus += hiddenBefore == 0 ? 3600 : 1200;
        } else {
            // Risky pushes remain searchable because occasionally a blocker
            // really must move.  The large debt cost prevents executing that
            // push as a short-sighted win; a later reveal/rebalance in the same
            // branch has to earn it back first.
            bonus -= (move.from < 0 ? 6000 : 9000) + std::max(0, debtDelta) * 3500;
        }
    }
    if (move.kind == VirtualMoveKind::FoundationToTableau) bonus -= 9000;
    // Exact talon/waste hashing catches loops. A fourth (and later) full
    // recycle is a score cost even after Tableau progress; the old physical
    // counter reset hid that from the planner. This is a preference, not a ban.
    if (move.kind == VirtualMoveKind::Talon) {
        bonus -= SOL_TALON_STEP_SEARCH_COST;
        if (before.talon.empty() && before.recycles >= SOL_FREE_TALON_RECYCLES)
            bonus -= SOL_CHARGED_RECYCLE_SEARCH_COST;
    }
    if (move.kind == VirtualMoveKind::Tableau && move.from >= 0 &&
        before.tableau[move.to].empty()) bonus += 450;
    return bonus;
}

static int scoreFutureCandidate(const GameState& current, const Move& candidate,
                                int deepSteps, int budget, uint64_t deadlineMs,
                                bool allowFoundationPullback = false,
                                bool allowDistantKing = false) {
    GameState after = current;
    if (!applyVirtualMove(after, candidate)) return -10000000;
    int repetitionPenalty = seenRecentSolverState(after, nowMs()) ? 4000000 : 0;
    PlannerSearchContext context;
    context.budget = budget;
    context.deadlineMs = deadlineMs;
    context.allowFoundationPullback = allowFoundationPullback;
    context.allowDistantKing = allowDistantKing;
    context.transposition.reserve((size_t)budget);
    context.path.reserve(64);
    const int future = searchVirtualFuture(after, std::clamp(deepSteps, 15, 20), context);
    if (future <= -8000000) return future;
    return future + candidateTacticalBonus(current, after, candidate) - repetitionPenalty;
}

static bool simulate_future_moves(GameState current, Move candidate_move, int deep_steps) {
    return scoreFutureCandidate(current, candidate_move, deep_steps, 720, nowMs() + 18) > -8000000;
}

static GameState makeGameState(const std::array<std::vector<Card>, 7>& tableau,
                               const std::array<std::vector<Card>, 4>& foundations,
                               const std::vector<Card>& talon,
                               const std::vector<Card>& waste) {
    GameState state;
    state.tableau = tableau;
    state.foundations = foundations;
    state.talon = talon;
    state.waste = waste;
    state.recycles = g_actualRecycleCount;
    state.stagnantRecycles = g_stagnantRecycleCount;
    state.drawCount = g_cardsPerTalonFlip;
    return state;
}

struct RescuePlan {
    bool valid = false;
    int from = -1;
    int to = -1;
    int cardId = -1;
    int score = -1000000;
};

static int tableauSearchScore(const std::array<std::vector<Card>, 7>& tableau,
                              const std::array<std::vector<Card>, 4>& foundations) {
    int hidden = 0;
    int exposed = 0;
    int empty = 0;
    bool kingAvailable = false;
    for (const auto& column : tableau) {
        if (column.empty()) { ++empty; continue; }
        for (const Card& card : column) {
            if (card.faceUp()) {
                ++exposed;
                if (cardRank(card) == 12) kingAvailable = true;
            } else {
                ++hidden;
            }
        }
    }
    const int usefulEmpty = kingAvailable ? empty : 0;
    return -hidden * 10000 + countProductiveOptions(tableau, foundations) * 1000 +
           exposed * 20 + usefulEmpty * 300;
}

static int searchTableauDepth(const std::array<std::vector<Card>, 7>& tableau,
                              const std::array<std::vector<Card>, 4>& foundations,
                              int depth, int& budget, int lastFrom, int lastTo, int lastCardId) {
    int best = tableauSearchScore(tableau, foundations);
    if (depth <= 0 || budget <= 0) return best;
    for (int from = 0; from < 7 && budget > 0; ++from) {
        const int first = firstFaceUpIndex(tableau[from]);
        if (first < 0) continue;
        for (int start = first; start < (int)tableau[from].size() && budget > 0; ++start) {
            const Card& moving = tableau[from][start];
            if (!moving.faceUp()) continue;
            for (int to = 0; to < 7 && budget > 0; ++to) {
                if (from == to) continue;
                if (moving.id == lastCardId && from == lastTo && to == lastFrom) continue;
                if (!isStrategicTableauMove(tableau, from, start, to, moving)) continue;
                --budget;
                auto next = tableau;
                std::vector<Card> sequence(next[from].begin() + start, next[from].end());
                next[from].erase(next[from].begin() + start, next[from].end());
                next[to].insert(next[to].end(), sequence.begin(), sequence.end());
                best = std::max(best, searchTableauDepth(next, foundations, depth - 1,
                                                         budget, from, to, moving.id));
            }
        }
    }
    return best;
}

static bool findRescuePlan(const std::array<std::vector<Card>, 7>& tableau,
                           const std::array<std::vector<Card>, 4>& foundations,
                           RescuePlan& out) {
    const int baseline = tableauSearchScore(tableau, foundations);
    int budget = 384; // bounded deep search; keeps Unity responsive.
    for (int from = 0; from < 7 && budget > 0; ++from) {
        const int first = firstFaceUpIndex(tableau[from]);
        if (first < 0) continue;
        for (int start = first; start < (int)tableau[from].size() && budget > 0; ++start) {
            const Card& moving = tableau[from][start];
            if (!moving.faceUp()) continue;
            for (int to = 0; to < 7 && budget > 0; ++to) {
                if (from == to) continue;
                if (!isStrategicTableauMove(tableau, from, start, to, moving)) continue;
                --budget;
                auto next = tableau;
                std::vector<Card> sequence(next[from].begin() + start, next[from].end());
                next[from].erase(next[from].begin() + start, next[from].end());
                next[to].insert(next[to].end(), sequence.begin(), sequence.end());
                const int score = searchTableauDepth(next, foundations, 4, budget,
                                                     from, to, moving.id);
                if ((!out.valid || score > out.score) && score > baseline) {
                    out = {true, from, to, moving.id, score};
                }
            }
        }
    }
    return out.valid;
}

static int nextUsefulTalonDistance(const std::vector<Card>& talon,
                                   const std::array<std::vector<Card>, 7>& tableau,
                                   const std::array<std::vector<Card>, 4>& foundations) {
    for (int i = (int)talon.size() - 1, distance = 0; i >= 0; --i, ++distance) {
        const Card& card = talon[i];
        if (!card.faceUp()) continue;
        for (int f = 0; f < 4; ++f) {
            const Card* top = foundations[f].empty() ? nullptr : &foundations[f].back();
            if (canDropOnFoundation(card, top) && foundationAdvanceIsSafe(card, foundations)) return distance;
        }
        for (int dst = 0; dst < 7; ++dst) {
            const Card* top = tableau[dst].empty() ? nullptr : &tableau[dst].back();
            if (canDropOnTableau(card, top)) return distance;
        }
    }
    return -1;
}

static void* findStackList(void* localGs, const std::string& name) {
    if (!localGs) return nullptr;
    void* dict = *(void**)((uint8_t*)localGs + OFF_SOL_STACKS);
    if (!dict) return nullptr;
    const int count = *(int*)((uint8_t*)dict + 0x20);
    void* entries = *(void**)((uint8_t*)dict + 0x18);
    if (!entries || count < 0 || count > 64) return nullptr;
    for (int i = 0; i < count; ++i) {
        const uint8_t* e = (const uint8_t*)entries + OFF_ARRAY_ITEMS + i * 24;
        if (*(const int*)e < 0) continue;
        if (readIl2CppString(*(void* const*)(e + 8)) == name) {
            return *(void* const*)(e + 16);
        }
    }
    return nullptr;
}

static void logCardRaw(const char* stack, int index, void* item) {
    void* values = item ? *(void**)((uint8_t*)item + OFF_JARRAY_VALUES) : nullptr;
    const int valueCount = listSize(values);
    void* first = values ? listItem(values, 0) : nullptr;
    void* second = values ? listItem(values, 1) : nullptr;
    void* third = values ? listItem(values, 2) : nullptr;
    const void* firstBox = first ? *(void* const*)((uint8_t*)first + OFF_JVALUE_VALUE) : nullptr;
    const void* secondBox = second ? *(void* const*)((uint8_t*)second + OFF_JVALUE_VALUE) : nullptr;
    const void* thirdBox = third ? *(void* const*)((uint8_t*)third + OFF_JVALUE_VALUE) : nullptr;
    const long long firstRaw = firstBox ? *(const long long*)((const uint8_t*)firstBox + 0x10) : -1;
    const long long secondRaw = secondBox ? *(const long long*)((const uint8_t*)secondBox + 0x10) : -1;
    Card parsed = parseCard(item, true);
    std::string props;
    void* collection = item ? *(void**)((uint8_t*)item + OFF_JOBJECT_PROPERTIES) : nullptr;
    void* propertyItems = collection ? *(void**)((uint8_t*)collection + OFF_COLLECTION_ITEMS) : nullptr;
    const int propertyCount = listSize(propertyItems);
    for (int i = 0; i < propertyCount; ++i) {
        void* property = listItem(propertyItems, i);
        if (!property) continue;
        const std::string key = readIl2CppString(*(void**)((uint8_t*)property + OFF_JPROPERTY_NAME));
        void* content = *(void**)((uint8_t*)property + OFF_JPROPERTY_CONTENT);
        void* value = content ? *(void**)((uint8_t*)content + OFF_JPROPERTYLIST_TOKEN) : nullptr;
        const int type = value ? *(int*)((uint8_t*)value + 0x30) : -1;
        void* box = value ? *(void**)((uint8_t*)value + OFF_JVALUE_VALUE) : nullptr;
        const long long raw = box ? *(const long long*)((uint8_t*)box + 0x10) : -1;
        if (!props.empty()) props += ";";
        props += key + "=" + std::to_string(raw) + ":t" + std::to_string(type);
        if (type == 8) props += ":" + tokenString(value);
    }
    void* exc = nullptr;
    void* serial = (item && g_objectToString) ? g_objectToString(item, &exc) : nullptr;
    const char* className = (item && g_classGetName) ? g_classGetName(*(void**)item) : "?";
    const std::string json = (!exc && serial) ? readIl2CppString(serial) : "<exception>";
    SOL_LOG("snapshot stack=%s idx=%d item=%p class=%s values=%p n=%d fields=[%p:t%d:box%p:v%lld,%p:t%d:box%p:v%lld,%p:t%d:box%p] props=%s parsed=id%d val%d color=%s json=%s",
            stack, index, item, className ? className : "?", values, valueCount,
            first, first ? *(int*)((uint8_t*)first + 0x30) : -1, firstBox, firstRaw,
            second, second ? *(int*)((uint8_t*)second + 0x30) : -1, secondBox, secondRaw,
            third, third ? *(int*)((uint8_t*)third + 0x30) : -1, thirdBox, props.c_str(),
            parsed.id, parsed.value, parsed.color.c_str(), json.c_str());
}

static void logFullSnapshot(void* localGs) {
    if (!localGs || g_loggedSnapshot) return;
    g_loggedSnapshot = true;
    SOL_LOG("snapshot begin prefix=%s", g_stackPrefix.c_str());
    for (int i = 0; i < 7; ++i) {
        const std::string name = g_stackPrefix + "tableau_" + std::to_string(i);
        void* list = findStackList(localGs, name);
        const int count = listSize(list);
        SOL_LOG("snapshot list=%s ptr=%p count=%d", name.c_str(), list, count);
        for (int j = 0; j < count; ++j) logCardRaw(name.c_str(), j, listItem(list, j));
    }
    for (const char* suffix : {"waste", "talon", "foundation_0", "foundation_1", "foundation_2", "foundation_3"}) {
        const std::string name = g_stackPrefix + suffix;
        void* list = findStackList(localGs, name);
        const int count = listSize(list);
        SOL_LOG("snapshot list=%s ptr=%p count=%d", name.c_str(), list, count);
        for (int j = 0; j < count; ++j) logCardRaw(name.c_str(), j, listItem(list, j));
    }
    SOL_LOG("snapshot end");
}

static std::vector<Card> stackCardsFromList(void* list) {
    std::vector<Card> out;
    const int count = listSize(list);
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        Card c = parseCard(listItem(list, i));
        if (c.valid()) out.push_back(std::move(c));
    }
    return out;
}

static void discoverStackPrefix(void* localGs) {
    if (!localGs) return;
    void* dict = *(void**)((uint8_t*)localGs + OFF_SOL_STACKS);
    if (!dict) return;
    const int count = *(int*)((uint8_t*)dict + 0x20);
    void* entries = *(void**)((uint8_t*)dict + 0x18);
    if (!entries || count < 0 || count > 64) return;
    // In a match LocalGs.Stacks holds both players' piles, so the first
    // "tableau_0" key can be the opponent's. Prefer the game's own index
    // (verified live: our piles are "p_<GetMyIndex>_"); fall back to the
    // first prefix when the index is unavailable or names no pile.
    // GetMyIndex falls back to 0 before the match state is complete, so
    // refresh it once a second rather than trusting the first answer.
    const uint64_t now = nowMs();
    if (g_getMyIndex && (g_myPlayerIndex < 0 || now - g_myPlayerIndexAt >= 1000)) {
        g_myPlayerIndex = g_getMyIndex(nullptr);
        g_myPlayerIndexAt = now;
    }
    const std::string mine = g_myPlayerIndex >= 0
        ? "p_" + std::to_string(g_myPlayerIndex) + "_" : std::string();
    std::string chosen;
    for (int i = 0; i < count; ++i) {
        const uint8_t* e = (const uint8_t*)entries + OFF_ARRAY_ITEMS + i * 24;
        if (*(const int*)e < 0) continue;
        const std::string key = readIl2CppString(*(void* const*)(e + 8));
        const size_t pos = key.find("tableau_0");
        if (pos == std::string::npos || pos == 0) continue;
        const std::string prefix = key.substr(0, pos);
        if (chosen.empty()) chosen = prefix;
        if (mine.empty() || prefix == mine) {
            chosen = prefix;
            break;
        }
    }
    if (!chosen.empty() && chosen != g_stackPrefix) {
        g_stackPrefix = chosen;
        g_columnVisibilityValid = false;   // the history described other piles
        SOL_LOG("using stack prefix=%s myIndex=%d", g_stackPrefix.c_str(), g_myPlayerIndex);
    }
}

// Lists can be replaced after each move, so resolve them afresh every tick.
// One dictionary pass avoids repeating the same key decoding for each stack.
struct SolitaireStackLists {
    std::array<void*, 7> tableau{};
    std::array<void*, 4> foundations{};
    void* waste = nullptr;
    void* talon = nullptr;
};

static SolitaireStackLists currentStackLists(void* localGs) {
    SolitaireStackLists result;
    if (!localGs) return result;
    void* dict = *(void**)((uint8_t*)localGs + OFF_SOL_STACKS);
    if (!dict) return result;
    const int count = *(int*)((uint8_t*)dict + 0x20);
    void* entries = *(void**)((uint8_t*)dict + 0x18);
    if (!entries || count < 0 || count > 64) return result;
    const size_t prefixLength = g_stackPrefix.size();
    for (int i = 0; i < count; ++i) {
        const uint8_t* entry = (const uint8_t*)entries + OFF_ARRAY_ITEMS + i * 24;
        if (*(const int*)entry < 0) continue;
        const std::string key = readIl2CppString(*(void* const*)(entry + 8));
        if (key.size() < prefixLength ||
            key.compare(0, prefixLength, g_stackPrefix) != 0) continue;
        const std::string_view suffix(key.data() + prefixLength,
                                      key.size() - prefixLength);
        void* list = *(void* const*)(entry + 16);
        if (suffix == "waste") result.waste = list;
        else if (suffix == "talon") result.talon = list;
        else if (suffix.size() == 9 && suffix.substr(0, 8) == "tableau_" &&
                 suffix[8] >= '0' && suffix[8] <= '6') {
            result.tableau[suffix[8] - '0'] = list;
        } else if (suffix.size() == 12 && suffix.substr(0, 11) == "foundation_" &&
                   suffix[11] >= '0' && suffix[11] <= '3') {
            result.foundations[suffix[11] - '0'] = list;
        }
    }
    return result;
}

static void logNativeStackKeys(void* localGs) {
    if (!localGs || g_loggedStackKeys) return;
    void* dict = *(void**)((uint8_t*)localGs + OFF_SOL_STACKS);
    if (!dict) return;
    // Dictionary<TKey,TValue>: entries at +0x18, count at +0x20.  An entry
    // with reference-type key/value is {hash,next,key,value}, 24 bytes.
    const int count = *(int*)((uint8_t*)dict + 0x20);
    void* entries = *(void**)((uint8_t*)dict + 0x18);
    if (!entries || count < 0 || count > 64) return;
    std::string keys;
    for (int i = 0; i < count; ++i) {
        const uint8_t* e = (const uint8_t*)entries + OFF_ARRAY_ITEMS + i * 24;
        const int hash = *(const int*)e;
        if (hash < 0) continue;
        std::string key = readIl2CppString(*(void* const*)(e + 8));
        if (key.empty()) continue;
        if (!keys.empty()) keys += ",";
        keys += key;
    }
    if (!keys.empty()) {
        g_loggedStackKeys = true;
        SOL_LOG("native stack keys=%s", keys.c_str());
    }
}

static void* findRaycastController() {
    if (g_rayController) return g_rayController;
    if (!g_domainGet || !g_domainAssemblies || !g_assemblyImage || !g_imageName ||
        !g_classFromName || !g_staticData) return nullptr;
    void* domain = g_domainGet();
    size_t count = 0;
    const void** assemblies = domain ? g_domainAssemblies(domain, &count) : nullptr;
    if (!assemblies || count > 256) return nullptr;
    for (size_t i = 0; i < count; ++i) {
        void* image = g_assemblyImage(assemblies[i]);
        const char* imageName = image ? g_imageName(image) : nullptr;
        if (!imageName || std::string(imageName) != "Assembly-CSharp.dll") continue;
        void* klass = g_classFromName(image, "Jawaker", "SolitaireRaycastController");
        void* statics = klass ? g_staticData(klass) : nullptr;
        void* instance = statics ? *(void**)statics : nullptr; // static Ins is the first field
        if (instance) {
            g_rayController = instance;
            SOL_LOG("raycast controller=%p", g_rayController);
        }
        return g_rayController;
    }
    return nullptr;
}

static void* findCardsController() {
    if (g_cardsController) return g_cardsController;
    if (!g_domainGet || !g_domainAssemblies || !g_assemblyImage || !g_imageName ||
        !g_classFromName || !g_staticData) return nullptr;
    void* domain = g_domainGet();
    size_t count = 0;
    const void** assemblies = domain ? g_domainAssemblies(domain, &count) : nullptr;
    if (!assemblies || count > 256) return nullptr;
    for (size_t i = 0; i < count; ++i) {
        void* image = g_assemblyImage(assemblies[i]);
        const char* imageName = image ? g_imageName(image) : nullptr;
        if (!imageName || std::string(imageName) != "Assembly-CSharp.dll") continue;
        void* klass = g_classFromName(image, "Jawaker", "CardsController");
        void* statics = klass ? g_staticData(klass) : nullptr;
        // CardsController.ins is its first and only relevant static field.
        g_cardsController = statics ? *(void**)statics : nullptr;
        if (g_cardsController) SOL_LOG("cards controller=%p", g_cardsController);
        return g_cardsController;
    }
    return nullptr;
}

static void* currentRawGameStoreState() {
    if (!g_gameStoreStatics) {
        if (!g_domainGet || !g_domainAssemblies || !g_assemblyImage ||
            !g_imageName || !g_classFromName || !g_staticData) return nullptr;
        void* domain = g_domainGet();
        size_t count = 0;
        const void** assemblies = domain ? g_domainAssemblies(domain, &count) : nullptr;
        if (!assemblies || count > 256) return nullptr;
        for (size_t i = 0; i < count; ++i) {
            void* image = g_assemblyImage(assemblies[i]);
            const char* imageName = image ? g_imageName(image) : nullptr;
            if (!imageName || std::string(imageName) != "Assembly-CSharp.dll")
                continue;
            void* klass = g_classFromName(image, "Jawaker.Games.Core", "GameStore");
            g_gameStoreStatics = klass ? g_staticData(klass) : nullptr;
            break;
        }
    }
    return g_gameStoreStatics
        ? *(void**)((uint8_t*)g_gameStoreStatics + OFF_GAMESTORE_STATIC_GS)
        : nullptr;
}

static void* findCardController(void* rayController, int cardId,
                                int expectedValue) {
    if (!rayController || cardId < 0 || expectedValue < 0 ||
        expectedValue >= 52) return nullptr;
    // SolitaireRaycastController._cardsPile is normally populated by the
    // multiplayer scene. Solo Solitaire usually uses the shared allCards
    // list; during a scene transition the ray list can also be nonempty but
    // incomplete, so a miss there must still try the shared list.
    void* rayCards = *(void**)((uint8_t*)rayController + 0xA0);
    const auto scan = [cardId, expectedValue](void* cards) -> void* {
        for (int i = 0, count = listSize(cards); i < count; ++i) {
            void* controller = listItem(cards, i);
            if (controller && *(int*)((uint8_t*)controller + 0x4C) == cardId &&
                *(int*)((uint8_t*)controller + 0x50) == expectedValue)
                return controller;
        }
        return nullptr;
    };
    if (void* found = scan(rayCards)) return found;
    void* cardsController = findCardsController();
    void* sharedCards = cardsController
        ? *(void**)((uint8_t*)cardsController + 0x60) : nullptr;
    if (sharedCards != rayCards) {
        if (void* found = scan(sharedCards)) return found;
    }
    const uint64_t now = nowMs();
    if (now - g_lastCardDiagMs > 1800) {
        g_lastCardDiagMs = now;
        void* visibleCards = sharedCards ? sharedCards : rayCards;
        const int count = listSize(visibleCards);
        std::string visible;
        for (int i = 0; i < count && i < 18; ++i) {
            void* controller = listItem(visibleCards, i);
            if (!controller) continue;
            if (!visible.empty()) visible += ',';
            visible += std::to_string(*(int*)((uint8_t*)controller + 0x4C));
            visible += ':';
            visible += std::to_string(*(int*)((uint8_t*)controller + 0x50));
        }
        SOL_LOG("card-view miss wanted=%d value=%d rayCount=%d sharedCount=%d visible=[%s]",
                cardId, expectedValue, listSize(rayCards), listSize(sharedCards),
                visible.c_str());
    }
    return nullptr;
}

static int activeSolitairePlayerIndex() {
    const size_t prefix = g_stackPrefix.find("p_");
    if (prefix == std::string::npos) return -1;
    int value = 0;
    bool found = false;
    for (size_t i = prefix + 2; i < g_stackPrefix.size(); ++i) {
        const char c = g_stackPrefix[i];
        if (c < '0' || c > '9') break;
        found = true;
        value = value * 10 + (c - '0');
    }
    return found ? value : -1;
}

// Opponent reveal leak (IDA, 28.3.16): SyncStacks runs
// UpdateRevealedHiddenCards on every player's piles, and RevealedCardIds is
// filled from every player's face-up cards (InitializeRevealedCardIds). Any of
// our face-down cards whose id an opponent has turned up therefore gets our
// mapped "val" written into LocalGs while it is still face down for us.
// Reading "has a value" as "face up" made the planner move cards the server
// still holds face down; each such move timed out (2.2 s) and was recorded as
// a failed edge until no executable plan was left and Auto froze (toggling it
// cleared those caches). A fast opponent bot triggers this constantly.
//
// Face-down cards always form the bottom of a column, and their count only
// drops when everything above the topmost one has left the column (the game
// then turns it up). Track that count across snapshots and keep any leaked
// value as knowledge (knownValue) on a card that stays face down.
static bool continuesRun(const Card& lower, const Card& upper) {
    return lower.faceUp() && upper.faceUp() &&
           cardRank(lower) - cardRank(upper) == 1 &&
           (cardSuit(lower) & 1) != (cardSuit(upper) & 1);
}

static void normalizeTableauVisibility(std::array<std::vector<Card>, 7>& tableau,
                                       bool commitHistory) {
    int masked = 0;
    for (int c = 0; c < 7; ++c) {
        std::vector<Card>& column = tableau[c];
        ColumnVisibility& seen = g_columnVisibility[c];
        const int size = (int)column.size();
        // A card sent without a value is certainly face down, and so is
        // everything below it.
        int floor = 0;
        for (int i = 0; i < size; ++i) if (!column[i].faceUp()) floor = i + 1;
        int hidden;
        if (!g_columnVisibilityValid) {
            // No history (round start, Auto just enabled): the face-up part
            // is the legal descending run that ends at the top card.
            hidden = size > 0 ? size - 1 : 0;
            while (hidden > floor && continuesRun(column[hidden - 1], column[hidden])) --hidden;
        } else {
            int common = 0;
            const int limit = std::min(size, (int)seen.ids.size());
            while (common < limit && seen.ids[(size_t)common] == column[(size_t)common].id) ++common;
            hidden = seen.hidden;
            // Everything from `common` up left the column at some point, so the
            // card below it became the top and was turned up (possibly more
            // than once between two snapshots, e.g. with queued moves).
            if (common <= hidden) hidden = std::max(0, common - 1);
        }
        hidden = std::max(hidden, floor);
        if (floor < size) hidden = std::min(hidden, size - 1);  // a valued top card is up
        hidden = std::min(hidden, size);
        for (int i = 0; i < hidden; ++i) {
            Card& card = column[(size_t)i];
            if (!card.faceUp()) continue;
            card.knownValue = card.value;
            card.value = -1;
            ++masked;
        }
        if (commitHistory) {
            seen.hidden = hidden;
            seen.ids.clear();
            for (const Card& card : column) seen.ids.push_back(card.id);
        }
    }
    if (!commitHistory) return;
    g_columnVisibilityValid = true;
    if (masked != g_leakMaskedCards) {
        SOL_LOG("[REVEAL_LEAK] face-down cards carrying a leaked value=%d (was %d)",
                masked, g_leakMaskedCards);
        g_leakMaskedCards = masked;
    }
}

static void hydratePerfectInformation(void* localGs,
                                      std::array<std::vector<Card>, 7>& tableau,
                                      std::vector<Card>& talon,
                                      std::vector<Card>& waste,
                                      std::array<std::vector<Card>, 4>& foundations) {
    std::unordered_map<int, int> observed;
    observed.reserve(52);
    const auto observe = [&observed](const std::vector<Card>& pile) {
        for (const Card& card : pile)
            if (card.id >= 0 && card.faceUp()) observed[card.id] = card.value;
    };
    for (const auto& column : tableau) observe(column);
    for (const auto& pile : foundations) observe(pile);
    observe(talon);
    observe(waste);

    // Once a complete deal has passed the 52-card bijection and live-visible
    // checks, it is immutable for this LocalGs round. Hydrate directly from the
    // compact array on subsequent ticks instead of walking IL2CPP dictionaries,
    // JSON objects and card controllers every frame. This is the hot-path speed
    // win: perfect information stays identical while per-move allocation and
    // reflection overhead disappears.
    const auto hydrateFromRoundCache = [](std::vector<Card>& pile) {
        for (Card& card : pile) {
            if (card.id >= 0 && card.id < 52) {
                const int value = g_roundCardValues[(size_t)card.id];
                if (value >= 0 && value < 52) card.knownValue = value;
                else if (card.faceUp()) card.knownValue = card.value;
            } else if (card.faceUp()) {
                card.knownValue = card.value;
            }
        }
    };
    if (g_mappingValidated) {
        bool visibleStillMatches = true;
        for (const auto& visible : observed) {
            if (visible.first < 0 || visible.first >= 52 ||
                g_roundCardValues[(size_t)visible.first] != visible.second) {
                visibleStillMatches = false;
                break;
            }
        }
        if (visibleStillMatches) {
            for (auto& column : tableau) hydrateFromRoundCache(column);
            for (auto& pile : foundations) hydrateFromRoundCache(pile);
            hydrateFromRoundCache(talon);
            hydrateFromRoundCache(waste);
            return;
        }

        // A visible conflict means Unity reused an object unexpectedly. Never
        // keep driving the solver with stale perfect information: retain only
        // values proven by the physical board and rebuild the deal mapping.
        g_mappingValidated = false;
        for (int id = 0; id < 52; ++id) {
            const auto it = observed.find(id);
            g_roundCardValues[(size_t)id] =
                it == observed.end() ? -1 : it->second;
        }
        SOL_LOG("perfect cache invalidated reason=visible-conflict");
    }

    struct MappingCandidate {
        std::vector<std::pair<int, int>> entries;
        int player = -1;
        int directMatches = 0;
        int directConflicts = 0;
        int reverseMatches = 0;
        bool reverse = false;
        bool complete = false;
        bool fromGameStore = false;
        bool mergedPlayers = false;
    };
    int players = 0;
    const int preferredPlayer = activeSolitairePlayerIndex();
    const auto isCompleteDirectMapping = [&observed](MappingCandidate& candidate) {
        if (candidate.entries.size() != 52 || candidate.directConflicts != 0)
            return false;
        std::array<bool, 52> seenIds{};
        std::array<bool, 52> seenValues{};
        for (const auto& entry : candidate.entries) {
            const int id = entry.first;
            const int value = entry.second;
            if (id < 0 || id >= 52 || value < 0 || value >= 52 ||
                seenIds[(size_t)id] || seenValues[(size_t)value])
                return false;
            seenIds[(size_t)id] = true;
            seenValues[(size_t)value] = true;
        }
        // A stale mapping from a previous deal (or another player) can still
        // be a 52-card permutation. Every physically visible card must agree.
        for (const auto& visible : observed) {
            bool matched = false;
            for (const auto& mapped : candidate.entries) {
                if (mapped.first == visible.first) {
                    matched = mapped.second == visible.second;
                    break;
                }
            }
            if (!matched) return false;
        }
        return true;
    };
    const auto mappingScore = [preferredPlayer](const MappingCandidate& candidate) {
        return (candidate.complete ? 1000000 : 0) +
               candidate.directMatches * 32 -
               candidate.directConflicts * 10000 +
               (int)candidate.entries.size() * 2 +
               (candidate.mergedPlayers ? 512 : 0) +
               (candidate.player == preferredPlayer ? 16 : 0);
    };

    const auto readBestMapping = [&](int& playerCount) {
        MappingCandidate best;
        void* mappings = localGs
            ? *(void**)((uint8_t*)localGs + OFF_SOL_PLAYER_CARDS_MAPPING)
            : nullptr;
        if (!mappings) return best;
        const size_t length = *(size_t*)((uint8_t*)mappings + 0x18);
        if (length == 0 || length > 8) return best;
        playerCount = (int)length;
        int bestScore = -100000000;
        std::array<int, 52> mergedValues{};
        mergedValues.fill(-1);
        bool mergedConflict = false;
        for (int player = 0; player < playerCount; ++player) {
            void* dictionary = *(void**)((uint8_t*)mappings + OFF_ARRAY_ITEMS +
                                         (size_t)player * sizeof(void*));
            if (!dictionary) continue;
            void* entriesArray = *(void**)((uint8_t*)dictionary + 0x18);
            const int count = *(int*)((uint8_t*)dictionary + 0x20);
            if (!entriesArray || count <= 0 || count > 128) continue;
            const size_t capacity = *(size_t*)((uint8_t*)entriesArray + 0x18);
            if (capacity == 0 || capacity > 128) continue;

            MappingCandidate candidate;
            candidate.player = player;
            candidate.entries.reserve((size_t)count);
            const int scan = std::min(count, (int)capacity);
            const uint8_t* base = (const uint8_t*)entriesArray + OFF_ARRAY_ITEMS;
            for (int i = 0; i < scan; ++i) {
                // IDA ground truth: SyncPlayerCardsMapping stores cardId as
                // the Dictionary key and cardValue as its value. No heuristic
                // key/value reversal is permitted here.
                const int* entry = (const int*)(base + (size_t)i * 16);
                if (entry[0] < 0 || entry[2] < 0 || entry[2] >= 52 ||
                    entry[3] < 0 || entry[3] >= 52) continue;
                candidate.entries.emplace_back(entry[2], entry[3]);
                int& merged = mergedValues[(size_t)entry[2]];
                if (merged < 0) merged = entry[3];
                else if (merged != entry[3]) mergedConflict = true;
                const auto visible = observed.find(entry[2]);
                if (visible != observed.end()) {
                    if (visible->second == entry[3]) ++candidate.directMatches;
                    else ++candidate.directConflicts;
                }
            }
            candidate.complete = isCompleteDirectMapping(candidate);
            const int candidateScore = mappingScore(candidate);
            if (!candidate.entries.empty() && candidateScore > bestScore) {
                bestScore = candidateScore;
                best = std::move(candidate);
            }
        }
        // Fair-match players share the same card-id/value permutation, but
        // each side learns hidden values independently as it reveals cards.
        // Merge the conflict-free union so an opponent that is further ahead
        // can fill our still-opaque ids without discarding values only we know.
        if (!mergedConflict) {
            MappingCandidate merged;
            merged.player = -2;
            merged.mergedPlayers = true;
            merged.entries.reserve(52);
            for (int id = 0; id < 52; ++id) {
                const int value = mergedValues[(size_t)id];
                if (value < 0 || value >= 52) continue;
                merged.entries.emplace_back(id, value);
                const auto visible = observed.find(id);
                if (visible != observed.end()) {
                    if (visible->second == value) ++merged.directMatches;
                    else ++merged.directConflicts;
                }
            }
            merged.complete = isCompleteDirectMapping(merged);
            const int mergedScore = mappingScore(merged);
            if (!merged.entries.empty() && mergedScore > bestScore) {
                bestScore = mergedScore;
                best = std::move(merged);
            }
        }
        return best;
    };

    const auto readBestRawGameStoreMapping = [&](int& playerCount) {
        MappingCandidate best;
        void* rawGs = currentRawGameStoreState();
        void* mappingToken = rawGs
            ? objectFieldAny(rawGs, {"player_cards_mapping", "playerCardsMapping"})
            : nullptr;
        void* playerArrays = mappingToken
            ? *(void**)((uint8_t*)mappingToken + OFF_JARRAY_VALUES)
            : nullptr;
        const int count = listSize(playerArrays);
        if (count <= 0 || count > 8) return best;
        playerCount = count;
        int bestScore = -100000000;
        std::array<int, 52> mergedValues{};
        mergedValues.fill(-1);
        bool mergedConflict = false;
        for (int player = 0; player < count; ++player) {
            void* playerArray = listItem(playerArrays, player);
            void* values = playerArray
                ? *(void**)((uint8_t*)playerArray + OFF_JARRAY_VALUES)
                : nullptr;
            const int valueCount = listSize(values);
            if (valueCount <= 0 || valueCount > 52) continue;

            MappingCandidate candidate;
            candidate.player = player;
            candidate.fromGameStore = true;
            candidate.entries.reserve((size_t)valueCount);
            for (int id = 0; id < valueCount; ++id) {
                const int value = tokenInt(listItem(values, id));
                if (value < 0 || value >= 52) continue;
                candidate.entries.emplace_back(id, value);
                int& merged = mergedValues[(size_t)id];
                if (merged < 0) merged = value;
                else if (merged != value) mergedConflict = true;
                const auto visible = observed.find(id);
                if (visible != observed.end()) {
                    if (visible->second == value) ++candidate.directMatches;
                    else ++candidate.directConflicts;
                }
            }
            candidate.complete = isCompleteDirectMapping(candidate);
            const int candidateScore = mappingScore(candidate);
            if (!candidate.entries.empty() && candidateScore > bestScore) {
                bestScore = candidateScore;
                best = std::move(candidate);
            }
        }
        if (!mergedConflict) {
            MappingCandidate merged;
            merged.player = -2;
            merged.fromGameStore = true;
            merged.mergedPlayers = true;
            merged.entries.reserve(52);
            for (int id = 0; id < 52; ++id) {
                const int value = mergedValues[(size_t)id];
                if (value < 0 || value >= 52) continue;
                merged.entries.emplace_back(id, value);
                const auto visible = observed.find(id);
                if (visible != observed.end()) {
                    if (visible->second == value) ++merged.directMatches;
                    else ++merged.directConflicts;
                }
            }
            merged.complete = isCompleteDirectMapping(merged);
            const int mergedScore = mappingScore(merged);
            if (!merged.entries.empty() && mergedScore > bestScore) {
                bestScore = mergedScore;
                best = std::move(merged);
            }
        }
        return best;
    };

    MappingCandidate best = readBestMapping(players);
    // Live Frida capture against 28.2.90 proved that this JObject stores the
    // mapping values as JSON strings and keeps unrevealed ids at -1. It also
    // carries one mapping per fair-match player. Read and conflict-check their
    // union until a complete 0..51 bijection is validated; every physically
    // visible local card must still agree before the union can drive search.
    if (!best.complete && !g_mappingValidated) {
        int rawPlayers = 0;
        MappingCandidate raw = readBestRawGameStoreMapping(rawPlayers);
        if (!raw.entries.empty() &&
            (best.entries.empty() || mappingScore(raw) > mappingScore(best))) {
            best = std::move(raw);
            players = rawPlayers;
        }
    }

    // The UI can publish all 52 stacks one frame before it calls the native
    // LocalGs mapping sync. Ask the game's own narrow sync routine to refresh
    // only this field. Retries are round-bounded and cease permanently once a
    // validated 52/52 permutation is obtained, so normal move pacing and the
    // expert search budgets remain bit-for-bit unchanged.
    int physicalCards = (int)talon.size() + (int)waste.size();
    for (const auto& column : tableau) physicalCards += (int)column.size();
    for (const auto& pile : foundations) physicalCards += (int)pile.size();
    const uint64_t mappingNow = nowMs();
    if (!best.complete && !g_mappingValidated && physicalCards == 52 &&
        g_syncPlayerCardsMapping &&
        g_mappingSyncAttempts < SOL_MAPPING_SYNC_MAX_ATTEMPTS &&
        (g_lastMappingSyncMs == 0 ||
         mappingNow - g_lastMappingSyncMs >= SOL_MAPPING_SYNC_RETRY_MS)) {
        g_lastMappingSyncMs = mappingNow;
        ++g_mappingSyncAttempts;
        const bool changed = g_syncPlayerCardsMapping(localGs);
        players = 0;
        best = readBestMapping(players);
        if (g_mappingSyncAttempts == 1 || best.complete ||
            g_mappingSyncAttempts == SOL_MAPPING_SYNC_MAX_ATTEMPTS) {
            SOL_LOG("perfect native-sync attempt=%d changed=%d player=%d pairs=%d matches=%d conflicts=%d full=%d",
                    g_mappingSyncAttempts, changed ? 1 : 0, best.player,
                    (int)best.entries.size(), best.directMatches,
                    best.directConflicts, best.complete ? 1 : 0);
        }
    }

    // Preserve every authoritative value already seen in this round.  The
    // game's PlayerCardsMapping is allowed to be empty for several snapshots
    // (and, in some bot rounds, for the whole deal), so rebuilding this map
    // from scratch each tick silently discarded useful perfect information.
    for (const auto& entry : observed) {
        if (entry.first >= 0 && entry.first < 52 &&
            entry.second >= 0 && entry.second < 52)
            g_roundCardValues[(size_t)entry.first] = entry.second;
    }
    if (best.complete) {
        // A complete bijection that agrees with every visible card is the
        // authoritative deal. Copy it atomically into the round cache.
        for (const auto& entry : best.entries)
            g_roundCardValues[(size_t)entry.first] = entry.second;
        g_mappingValidated = true;
    } else if (best.directConflicts == 0) {
        // Partial snapshots remain useful, but they may only fill unknown ids;
        // they can never overwrite a value verified from the physical board.
        for (const auto& entry : best.entries) {
            const int id = entry.first;
            const int value = entry.second;
            if (id < 0 || id >= 52 || value < 0 || value >= 52) continue;
            int& cached = g_roundCardValues[(size_t)id];
            if (cached < 0 || cached == value) cached = value;
        }
    }

    // Ray._cardsPile can be nonempty yet omit controllers that the shared
    // CardsController.allCards list already owns. The physical dispatcher
    // checks both lists; hydration must do the same or the planner can remain
    // opaque despite a complete, validated deal being present in memory.
    // Inspect each source independently: the static shared list may still
    // belong to the previous round during a scene transition.
    void* ray = findRaycastController();
    void* rayCards = ray ? *(void**)((uint8_t*)ray + 0xA0) : nullptr;
    void* cardsController = findCardsController();
    void* sharedCards = cardsController
        ? *(void**)((uint8_t*)cardsController + 0x60) : nullptr;
    const int requiredControllerMatches = std::min(4, (int)observed.size());
    const auto acceptControllerList = [&](void* cards, const char* source) {
        const int count = listSize(cards);
        if (count <= 0 || count > 128) return;
        std::array<int, 52> controllerValues{};
        controllerValues.fill(-1);
        int conflicts = 0;
        for (int i = 0; i < count; ++i) {
            void* controller = listItem(cards, i);
            if (!controller) continue;
            const int id = *(int*)((uint8_t*)controller + 0x4C);
            const int value = *(int*)((uint8_t*)controller + 0x50);
            if (id < 0 || id >= 52 || value < 0 || value >= 52) continue;
            int& candidate = controllerValues[(size_t)id];
            if (candidate >= 0 && candidate != value) ++conflicts;
            else candidate = value;
        }
        int matches = 0;
        int available = 0;
        for (int id = 0; id < 52; ++id) {
            const int value = controllerValues[(size_t)id];
            if (value < 0) continue;
            ++available;
            const auto visible = observed.find(id);
            if (visible != observed.end()) {
                if (visible->second == value) ++matches;
                else ++conflicts;
            }
            const int cached = g_roundCardValues[(size_t)id];
            if (cached >= 0 && cached != value) ++conflicts;
        }
        // A stale singleton or a mixed-player list cannot establish this
        // round's hidden cards. Require agreement with multiple live faces.
        if (conflicts != 0 || matches < requiredControllerMatches) {
            if (!g_loggedMappingSnapshot)
                SOL_LOG("perfect controllers source=%s count=%d available=%d matches=%d conflicts=%d accepted=0",
                        source, count, available, matches, conflicts);
            return;
        }
        int added = 0;
        for (int id = 0; id < 52; ++id) {
            const int value = controllerValues[(size_t)id];
            int& cached = g_roundCardValues[(size_t)id];
            if (value >= 0 && cached < 0) {
                cached = value;
                ++added;
            }
        }
        if (!g_loggedMappingSnapshot || added > 0)
            SOL_LOG("perfect controllers source=%s count=%d available=%d matches=%d added=%d accepted=1",
                    source, count, available, matches, added);
    };
    acceptControllerList(rayCards, "ray");
    if (sharedCards != rayCards)
        acceptControllerList(sharedCards, "shared");

    // If every card except one has a distinct verified value, the remaining
    // id/value pair is fixed by the 52-card deck bijection. This often turns
    // a 51/52 late-game snapshot into exact information without guessing.
    std::array<bool, 52> knownValues{};
    int missingId = -1;
    int missingCount = 0;
    bool knownValuesUnique = true;
    for (int id = 0; id < 52; ++id) {
        const int value = g_roundCardValues[(size_t)id];
        if (value < 0 || value >= 52) {
            missingId = id;
            ++missingCount;
        } else {
            if (knownValues[(size_t)value]) knownValuesUnique = false;
            knownValues[(size_t)value] = true;
        }
    }
    if (missingCount == 1 && knownValuesUnique) {
        for (int value = 0; value < 52; ++value) {
            if (!knownValues[(size_t)value]) {
                g_roundCardValues[(size_t)missingId] = value;
                SOL_LOG("perfect cache inferred final id=%d value=%d", missingId, value);
                break;
            }
        }
    }

    // A 52-card cache may drive the long perfect-information horizon only when
    // it is a real deck permutation and still agrees with every visible card.
    // If a stale controller ever completes an invalid cache, discard only its
    // unverified values and transparently fall back to the proven V4 behavior.
    bool cacheComplete = true;
    bool cacheValid = true;
    std::array<bool, 52> cacheValues{};
    for (int id = 0; id < 52; ++id) {
        const int value = g_roundCardValues[(size_t)id];
        if (value < 0 || value >= 52) {
            cacheComplete = false;
            continue;
        }
        if (cacheValues[(size_t)value]) cacheValid = false;
        cacheValues[(size_t)value] = true;
        const auto visible = observed.find(id);
        if (visible != observed.end() && visible->second != value)
            cacheValid = false;
    }
    if (cacheComplete && cacheValid) {
        g_mappingValidated = true;
    } else if (cacheComplete) {
        for (int id = 0; id < 52; ++id)
            if (observed.find(id) == observed.end())
                g_roundCardValues[(size_t)id] = -1;
        SOL_LOG("perfect cache rejected reason=non-bijective-or-visible-conflict");
    }

    std::unordered_map<int, int> values;
    values.reserve(52);
    for (int id = 0; id < 52; ++id) {
        const int value = g_roundCardValues[(size_t)id];
        if (value >= 0 && value < 52) values[id] = value;
    }
    if (!g_loggedMappingSnapshot) {
        g_loggedMappingSnapshot = true;
        SOL_LOG("perfect mapping players=%d preferred=%d selected=%d source=%s pairs=%d direct=%d reverse=%d orientation=%s resolved=%d cached=%d validated=%d syncAttempts=%d",
                players, preferredPlayer, best.player,
                best.fromGameStore ? "gamestore" : "localgs",
                (int)best.entries.size(),
                best.directMatches, best.reverseMatches, best.reverse ? "reverse" : "direct",
                (int)observed.size(), (int)values.size(),
                g_mappingValidated ? 1 : 0, g_mappingSyncAttempts);
    }
    if ((int)values.size() > g_lastRoundKnownCount) {
        const uint64_t now = nowMs();
        const int previous = g_lastRoundKnownCount;
        g_lastRoundKnownCount = (int)values.size();
        if (g_lastRoundKnownCount == 52 || now - g_lastPerfectInfoDiagMs >= 1500) {
            g_lastPerfectInfoDiagMs = now;
            SOL_LOG("perfect cache advanced=%d->%d opaque=%d",
                    previous, g_lastRoundKnownCount, 52 - g_lastRoundKnownCount);
        }
    }
    const auto hydrate = [&values](std::vector<Card>& pile) {
        for (Card& card : pile) {
            const auto it = values.find(card.id);
            if (it != values.end()) card.knownValue = it->second;
            else if (card.faceUp()) card.knownValue = card.value;
        }
    };
    for (auto& column : tableau) hydrate(column);
    for (auto& pile : foundations) hydrate(pile);
    hydrate(talon);
    hydrate(waste);
}

static std::string uiStackName(const std::string& stack) {
    // The solo LocalGs owns player-qualified stack keys (p_1_tableau_…).
    // MoveCardTo forwards these strings into the original action payload, so
    // stripping the qualifier makes the native action a no-op.
    return stack;
}

static bool reversesRecentMove(int cardId, const std::string& from,
                               const std::string& to, uint64_t now) {
    // A successful tableau shuffle can otherwise reappear a few seconds
    // later after another card moves. Keep a short tactical memory, not just
    // the immediately preceding edge.
    for (auto it = g_moveHistory.rbegin(); it != g_moveHistory.rend(); ++it) {
        if (now - it->at > 120000) break;
        if (it->cardId == cardId && it->from == to && it->to == from) return true;
    }
    return false;
}

static bool moveCard(void* localGs, const Card& card,
                     const std::string& from, const std::string& to,
                     int targetIndex) {
    g_lastDispatchOutcome = DispatchOutcome::Rejected;
    if (!card.faceUp() || !g_stringNew || !g_rayMove) return false;
    const uint64_t now = nowMs();
    if (!submissionQueueHasRoom(localGs)) {
        g_lastDispatchOutcome = DispatchOutcome::Deferred;
        g_dispatchDeferredThisTick = true;
        g_dispatchRetryAtMs = now + SOL_PIPELINE_RETRY_MS;
        return false;
    }
    if (card.id == g_lastMoveId && now - g_lastMoveAttemptMs < 1800 &&
        from == g_lastMoveFrom && to == g_lastMoveTo) {
        // Protect only against re-submitting the same asynchronous drag while
        // its animation is still settling. Strategic loop prevention belongs
        // to the exact GameState history, not to a long-lived card-id ban.
        g_lastDispatchOutcome = DispatchOutcome::Deferred;
        g_dispatchDeferredThisTick = true;
        g_dispatchRetryAtMs = g_lastMoveAttemptMs + 1800;
        return false;
    }
    void* ray = findRaycastController();
    void* controller = findCardController(ray, card.id, card.value);
    if (!controller) {
        g_lastDispatchOutcome = DispatchOutcome::Deferred;
        g_dispatchDeferredThisTick = true;
        g_dispatchRetryAtMs = now + SOL_CARD_VIEW_RETRY_MS;
        if (g_cardViewMissId != card.id || g_cardViewMissSinceMs == 0) {
            g_cardViewMissId = card.id;
            g_cardViewMissSinceMs = now;
        }
        // A scene can replace the static controller instance while the board
        // stays unchanged. Reacquire after sustained misses, but never call a
        // missing view a strategically invalid card move.
        if (now - g_cardViewMissSinceMs >= SOL_CARD_VIEW_REFRESH_MS &&
            now - g_lastCardViewRefreshMs >= SOL_CARD_VIEW_REFRESH_MS) {
            g_rayController = nullptr;
            g_cardsController = nullptr;
            g_lastCardViewRefreshMs = now;
        }
        if (!ray && now - g_lastCardDiagMs > 1800) {
            g_lastCardDiagMs = now;
            SOL_LOG("move wait ray-controller id=%d", card.id);
        }
        return false;
    }
    g_cardViewMissId = -1;
    g_cardViewMissSinceMs = 0;
    // GameRaycastController.cardController (0x58) is the exact card selected
    // by a physical drag. MoveCardTo then creates the game's normal DropCard.
    *(void**)((uint8_t*)ray + 0x58) = controller;
    const std::string target = uiStackName(to);
    const std::string source = uiStackName(from);
    // Match the game controller's own destination index: it is the current
    // number of cards in the destination stack (zero for an empty stack).
    if (targetIndex < 0 || targetIndex > 52) return false;
    const int64_t sequenceBefore = localSequenceNumber(localGs);
    g_accessStage.store(44, std::memory_order_release);
    SOL_LOG("[NATIVE_DISPATCH] card=%d target=%s from=%s index=%d",
            card.id, target.c_str(), source.c_str(), targetIndex);
    g_rayMove(ray, g_stringNew(target.c_str()), targetIndex,
              g_stringNew(source.c_str()), nullptr);
    SOL_LOG("[NATIVE_DISPATCH] returned card=%d", card.id);
    g_accessStage.store(4, std::memory_order_release);
    captureLastSubmissionObservation(localGs, sequenceBefore);
    g_lastDispatchOutcome = DispatchOutcome::Submitted;
    g_dispatchRetryPhysicalKey.clear();
    g_dispatchRetryAtMs = 0;
    g_lastActionWasTalon = false;
    g_lastMoveId = card.id;
    g_lastMoveFrom = from;
    g_lastMoveTo = to;
    g_lastMoveAttemptMs = now;
    g_moveHistory.push_back({card.id, from, to, now});
    if (g_moveHistory.size() > 48) g_moveHistory.erase(g_moveHistory.begin(), g_moveHistory.begin() + 16);
    SOL_LOG("ray-move id=%d val=%d from=%s to=%s", card.id, card.value, source.c_str(), target.c_str());
    return true;
}

static std::string tableauName(int index) {
    return g_stackPrefix + "tableau_" + std::to_string(index);
}

static std::string foundationName(int index) {
    return g_stackPrefix + "foundation_" + std::to_string(index);
}

static void rememberSolverState(const GameState& state, uint64_t now) {
    const std::string key = recentPositionKey(state);
    if (!g_solverStateHistory.empty() && g_solverStateHistory.back().tableau == key) return;
    g_solverStateHistory.push_back({key, now});
    while (g_solverStateHistory.size() > 96 ||
           (!g_solverStateHistory.empty() && now - g_solverStateHistory.front().at > 180000))
        g_solverStateHistory.erase(g_solverStateHistory.begin());
}

static void logPerfectPlannerSnapshot(const GameState& state) {
    if (g_loggedPerfectSnapshot) return;
    g_loggedPerfectSnapshot = true;
    const auto logPile = [](const char* name, const std::vector<Card>& pile) {
        std::string cards;
        cards.reserve(pile.size() * 10);
        for (const Card& card : pile) {
            if (!cards.empty()) cards.push_back(',');
            cards += std::to_string(card.id);
            cards.push_back(':');
            cards += std::to_string(card.known() ?
                                    (card.faceUp() ? card.value : card.knownValue) : -1);
            cards.push_back(card.faceUp() ? 'u' : 'd');
        }
        SOL_LOG("perfect %s=[%s]", name, cards.c_str());
    };
    SOL_LOG("perfect begin draw=%d", state.drawCount);
    for (int i = 0; i < 7; ++i) {
        const std::string name = "tableau_" + std::to_string(i);
        logPile(name.c_str(), state.tableau[i]);
    }
    logPile("talon", state.talon);
    logPile("waste", state.waste);
    for (int i = 0; i < 4; ++i) {
        const std::string name = "foundation_" + std::to_string(i);
        logPile(name.c_str(), state.foundations[i]);
    }
    SOL_LOG("perfect end");
}

static std::string moveKey(const Move& move) {
    return std::to_string((int)move.kind) + ':' + std::to_string(move.from) + ':' +
           std::to_string(move.to) + ':' + std::to_string(move.start);
}

// Search-only encoding. Keep Card/GameState and every Unity/JNI layout intact.
// One immutable table per root owns values/IDs/strings; nodes store only pile
// lengths, card-table indices and visibility bits (no heap-owning members).
struct SearchDeck {
    std::array<Card, 52> cards{};
    std::array<int8_t, 52> byId{};
    uint8_t count = 0;
    bool initialize(const GameState& state) {
        count = 0;
        byId.fill(-1);
        const auto add = [&](const std::vector<Card>& pile) {
            for (const Card& card : pile) {
                if (count == 52 || card.id < 0) return false;
                for (int i = 0; i < count; ++i)
                    if (cards[i].id == card.id) return false;
                cards[count] = card;
                // Revealing a known card changes visibility, not its identity.
                if (card.faceUp()) cards[count].knownValue = card.value;
                if (card.id < 52) byId[card.id] = (int8_t)count;
                ++count;
            }
            return true;
        };
        for (const auto& pile : state.tableau) if (!add(pile)) return false;
        for (const auto& pile : state.foundations) if (!add(pile)) return false;
        return add(state.talon) && add(state.waste);
    }
    int index(const Card& card) const {
        if (card.id >= 0 && card.id < 52) return byId[card.id];
        for (int i = 0; i < count; ++i) if (cards[i].id == card.id) return i;
        return -1;
    }
};

struct SearchBoard {
    std::array<uint8_t, 52> cards{};
    std::array<uint8_t, 13> sizes{};
    int recycles = 0;
    int stagnantRecycles = 0;
    int drawCount = 3;
};
static_assert(sizeof(SearchBoard) <= 80, "search board must remain compact");

static bool packSearchBoard(const GameState& state, const SearchDeck& deck,
                            SearchBoard& out) {
    size_t position = 0, pileIndex = 0;
    const auto pack = [&](const std::vector<Card>& pile) {
        if (pile.size() > 52 || position + pile.size() > 52) return false;
        out.sizes[pileIndex++] = (uint8_t)pile.size();
        for (const Card& card : pile) {
            const int index = deck.index(card);
            if (index < 0 || index >= deck.count) return false;
            const int value = card.faceUp() ? card.value : card.knownValue;
            if (value != deck.cards[index].knownValue) return false;
            out.cards[position++] = (uint8_t)(index |
                (card.faceUp() ? 0x40 : 0) | (card.virtuallyRevealed ? 0x80 : 0));
        }
        return true;
    };
    for (const auto& pile : state.tableau) if (!pack(pile)) return false;
    for (const auto& pile : state.foundations) if (!pack(pile)) return false;
    if (!pack(state.talon) || !pack(state.waste) || position != deck.count) return false;
    out.recycles = state.recycles;
    out.stagnantRecycles = state.stagnantRecycles;
    out.drawCount = state.drawCount;
    return true;
}

static void reserveSearchScratch(GameState& state) {
    for (auto& pile : state.tableau) pile.reserve(52);
    for (auto& pile : state.foundations) pile.reserve(13);
    state.talon.reserve(52);
    state.waste.reserve(52);
}

static bool unpackSearchBoard(const SearchBoard& board, const SearchDeck& deck,
                              GameState& state) {
    size_t position = 0, pileIndex = 0;
    const auto unpack = [&](std::vector<Card>& pile) {
        const size_t size = board.sizes[pileIndex++];
        if (position + size > deck.count) return false;
        pile.resize(size);
        for (Card& card : pile) {
            const uint8_t packed = board.cards[position++];
            const unsigned index = packed & 0x3f;
            if (index >= deck.count) return false;
            const Card& base = deck.cards[index];
            card.id = base.id;
            card.knownValue = base.knownValue;
            card.value = (packed & 0x40) ? base.knownValue : -1;
            card.virtuallyRevealed = (packed & 0x80) != 0;
            // Colour is a presentation field, unused by the native rules.
            card.color.clear();
        }
        return true;
    };
    for (auto& pile : state.tableau) if (!unpack(pile)) return false;
    for (auto& pile : state.foundations) if (!unpack(pile)) return false;
    if (!unpack(state.talon) || !unpack(state.waste) || position != deck.count) return false;
    state.recycles = board.recycles;
    state.stagnantRecycles = board.stagnantRecycles;
    state.drawCount = board.drawCount;
    return true;
}

// Root-local keys retain the complete position, not only a hash. Card-table
// indices are safe here because the immutable deck belongs to this root and
// is discarded whenever observed knowledge changes. Physical/global history
// continues to use its existing identity/knowledge-aware keys.
struct SearchPositionKey {
    std::array<uint8_t, 72> bytes{};
    bool operator==(const SearchPositionKey& other) const { return bytes == other.bytes; }
};
struct SearchPositionHash {
    size_t operator()(const SearchPositionKey& key) const {
        uint64_t hash = 14695981039346656037ULL;
        for (uint8_t byte : key.bytes) { hash ^= byte; hash *= 1099511628211ULL; }
        return (size_t)hash;
    }
};
static uint8_t canonicalSearchToken(uint8_t token) {
    // The existing solver key gives face-up precedence over virtual reveal.
    return token & 0x40 ? token & 0x7f : token;
}
static SearchPositionKey searchPositionKey(const SearchBoard& board, const SearchDeck& deck,
                                            bool stockVisibility = false) {
    SearchPositionKey key;
    std::array<unsigned, 14> offset{};
    for (int i = 0; i < 13; ++i) offset[i + 1] = offset[i] + board.sizes[i];
    std::array<int, 7> columns{0, 1, 2, 3, 4, 5, 6};
    std::sort(columns.begin(), columns.end(), [&](int a, int b) {
        if (board.sizes[a] != board.sizes[b]) return board.sizes[a] < board.sizes[b];
        return std::lexicographical_compare(board.cards.begin() + offset[a],
            board.cards.begin() + offset[a + 1], board.cards.begin() + offset[b],
            board.cards.begin() + offset[b + 1], [](uint8_t a, uint8_t b) {
                return canonicalSearchToken(a) < canonicalSearchToken(b);
            });
    });
    size_t cursor = 0;
    for (int col : columns) {
        key.bytes[cursor++] = board.sizes[col];
        for (unsigned i = offset[col]; i < offset[col + 1]; ++i)
            key.bytes[cursor++] = canonicalSearchToken(board.cards[i]);
    }
    std::array<uint8_t, 4> foundation{};
    for (int pile = 7; pile < 11; ++pile) {
        if (!board.sizes[pile]) continue;
        const Card& top = deck.cards[board.cards[offset[pile + 1] - 1] & 0x3f];
        const int suit = knownCardSuit(top), rank = knownCardRank(top);
        if (suit >= 0 && rank >= 0) foundation[suit] = (uint8_t)(rank + 1);
    }
    for (uint8_t rank : foundation) key.bytes[cursor++] = rank;
    for (int pile = 11; pile < 13; ++pile) {
        key.bytes[cursor++] = board.sizes[pile];
        for (unsigned i = offset[pile]; i < offset[pile + 1]; ++i)
            key.bytes[cursor++] = stockVisibility ? canonicalSearchToken(board.cards[i]) : board.cards[i] & 0x3f;
    }
    const uint32_t draw = (uint32_t)board.drawCount;
    for (int i = 0; i < 4; ++i) key.bytes[cursor++] = (uint8_t)(draw >> (i * 8));
    return key;
}
static uint32_t searchMoveId(const Move& move) {
    return (uint32_t)move.kind | ((uint32_t)(move.from + 1) << 8) |
           ((uint32_t)(move.to + 1) << 16) | ((uint32_t)(move.start + 1) << 24);
}
struct SearchVisitKey {
    SearchPositionKey position;
    uint32_t first = 0;
    bool widened = false;
    bool operator==(const SearchVisitKey& other) const {
        return first == other.first && widened == other.widened && position == other.position;
    }
};
struct SearchVisitHash {
    size_t operator()(const SearchVisitKey& key) const {
        return SearchPositionHash{}(key.position) ^ ((size_t)key.first * 2654435761ULL) ^
               (key.widened ? (size_t)0x9e3779b9U : 0);
    }
};

struct StockOrbitKey {
    std::array<uint8_t, 52> order{};
    uint8_t stock = 0, waste = 0;
    int drawCount = 3;
    bool operator==(const StockOrbitKey& other) const {
        return stock == other.stock && waste == other.waste &&
               drawCount == other.drawCount && order == other.order;
    }
};
struct StockOrbitHash {
    size_t operator()(const StockOrbitKey& key) const {
        uint64_t hash = 14695981039346656037ULL;
        for (uint8_t byte : key.order) { hash ^= byte; hash *= 1099511628211ULL; }
        return (size_t)(hash ^ ((uint64_t)key.stock << 8) ^ key.waste ^
                        ((uint64_t)(uint32_t)key.drawCount << 16));
    }
};
struct StockOrbitStep {
    StockOrbitKey position;
    std::array<uint8_t, 52> tokens{}; // exact visibility, unlike cycle equivalence
    uint8_t flips = 0, extraRecycles = 0;
};
using StockOrbitCache = std::unordered_map<StockOrbitKey, std::vector<StockOrbitStep>, StockOrbitHash>;
static StockOrbitKey stockOrbitKey(const SearchBoard& board, bool exactVisibility = false) {
    StockOrbitKey key;
    key.stock = board.sizes[11];
    key.waste = board.sizes[12];
    key.drawCount = board.drawCount;
    unsigned begin = 0;
    for (int pile = 0; pile < 11; ++pile) begin += board.sizes[pile];
    for (unsigned i = 0; i < key.stock + key.waste; ++i)
        key.order[i] = i < key.stock && !exactVisibility
            ? board.cards[begin + i] & 0x3f : board.cards[begin + i];
    return key;
}

// The same move rules on a byte board. Legality still calls the existing
// canDrop helpers; the only replacement is vector/Card movement. Differential
// replay tests compare every generated edge with applyVirtualMove.
static bool applySearchMove(SearchBoard& board, const SearchDeck& deck, const Move& move) {
    std::array<unsigned, 14> offset{};
    for (int i = 0; i < 13; ++i) offset[i + 1] = offset[i] + board.sizes[i];
    if (offset[13] != deck.count) return false;
    const auto decode = [&](uint8_t token) {
        Card card;
        const Card& base = deck.cards[token & 0x3f];
        card.id = base.id;
        card.knownValue = base.knownValue;
        card.value = token & 0x40 ? base.knownValue : -1;
        card.virtuallyRevealed = (token & 0x80) != 0;
        return card;
    };
    if (move.kind == VirtualMoveKind::Talon) {
        if (board.sizes[11]) {
            const int draw = std::min(std::clamp(board.drawCount, 1, 3), (int)board.sizes[11]);
            std::array<uint8_t, 3> drawn{};
            for (int i = 0; i < draw; ++i) {
                drawn[i] = board.cards[offset[11] + i];
                if (deck.cards[drawn[i] & 0x3f].known()) drawn[i] |= 0x40;
            }
            // Stock remainder precedes the old Waste, followed by drawn cards.
            std::move(board.cards.begin() + offset[11] + draw,
                      board.cards.begin() + offset[13], board.cards.begin() + offset[11]);
            std::copy_n(drawn.begin(), draw, board.cards.begin() + offset[13] - draw);
            board.sizes[11] -= draw;
            board.sizes[12] += draw;
        } else {
            if (!board.sizes[12]) return false;
            board.sizes[11] = board.sizes[12];
            board.sizes[12] = 0; // same order as the native virtual rules
            ++board.recycles;
            ++board.stagnantRecycles;
        }
        return true;
    }
    int source = -1, destination = -1, start = 0;
    if (move.kind == VirtualMoveKind::Foundation) {
        if (move.to < 0 || move.to >= 4 || move.from >= 7) return false;
        source = move.from < 0 ? 12 : move.from;
        destination = 7 + move.to;
        start = (int)board.sizes[source] - 1;
    } else if (move.kind == VirtualMoveKind::FoundationToTableau) {
        if (move.from < 0 || move.from >= 4 || move.to < 0 || move.to >= 7) return false;
        source = 7 + move.from;
        destination = move.to;
        start = (int)board.sizes[source] - 1;
    } else {
        if (move.to < 0 || move.to >= 7 || move.from >= 7 || move.from == move.to) return false;
        source = move.from < 0 ? 12 : move.from;
        destination = move.to;
        start = move.from < 0 ? (int)board.sizes[source] - 1 : move.start;
    }
    if (start < 0 || start >= board.sizes[source]) return false;
    const Card moving = decode(board.cards[offset[source] + start]);
    Card target;
    const Card* top = nullptr;
    if (board.sizes[destination]) {
        target = decode(board.cards[offset[destination + 1] - 1]);
        top = &target;
    }
    if (move.kind == VirtualMoveKind::Foundation) {
        if (!canDropOnFoundation(moving, top)) return false;
    } else {
        if (!canDropOnTableau(moving, top)) return false;
        if (move.kind == VirtualMoveKind::Tableau && move.from >= 0 && !top && start == 0)
            return false; // unchanged King-to-empty structural-loop rule
    }
    const int count = move.kind == VirtualMoveKind::Tableau && move.from >= 0
        ? (int)board.sizes[source] - start : 1;
    const auto previous = board.cards;
    unsigned cursor = 0;
    for (int pile = 0; pile < 13; ++pile) {
        const unsigned keep = board.sizes[pile] - (pile == source ? count : 0);
        std::copy_n(previous.begin() + offset[pile], keep, board.cards.begin() + cursor);
        cursor += keep;
        if (pile == destination) {
            std::copy_n(previous.begin() + offset[source] + start, count, board.cards.begin() + cursor);
            cursor += count;
        }
    }
    board.sizes[source] -= count;
    board.sizes[destination] += count;
    if (source < 7 && board.sizes[source]) {
        unsigned end = 0;
        for (int pile = 0; pile <= source; ++pile) end += board.sizes[pile];
        uint8_t& token = board.cards[end - 1];
        if (!(token & 0xc0)) {
            if (deck.cards[token & 0x3f].known()) token |= 0x40;
            else token |= 0x80;
        }
    }
    if (move.kind != VirtualMoveKind::FoundationToTableau) board.stagnantRecycles = 0;
    return true;
}

constexpr uint32_t SOL_NO_PARENT = std::numeric_limits<uint32_t>::max();
struct SearchPathLink {
    uint32_t parent = SOL_NO_PARENT;
    Move move;
    uint8_t stockFlips = 0; // these real actions precede move, never skipped
};

struct PersistentNode {
    SearchBoard board;
    uint32_t path = SOL_NO_PARENT;
    Move first;
    int depth = 0;
    int accumulated = 0;
    int hiddenCount = 0;
    int foundationCount = 0;
    bool autoCompleteProven = false;
    int structuralScore = -10000000;
    int priority = -10000000;
};

struct PersistentNodeLess {
    bool operator()(const PersistentNode& a, const PersistentNode& b) const {
        return a.priority < b.priority;
    }
};

class PersistentFrontier : public std::priority_queue<
        PersistentNode, std::vector<PersistentNode>, PersistentNodeLess> {
public:
    PersistentNode takeBalancedRoot(std::unordered_map<uint32_t, int>& coverage, bool& warming) {
        size_t selected = c.size();
        int least = 2;
        for (size_t i = 0; i < c.size(); ++i) {
            if (c[i].path == SOL_NO_PARENT) continue;
            const auto found = coverage.find(searchMoveId(c[i].first));
            if (found == coverage.end() || found->second >= 2) continue;
            if (found->second < least || (found->second == least &&
                (selected == c.size() || c[i].priority > c[selected].priority))) {
                selected = i;
                least = found->second;
            }
        }
        if (selected == c.size()) { warming = false; return takeTop(); }
        ++coverage[searchMoveId(c[selected].first)];
        PersistentNode result = std::move(c[selected]);
        if (selected == c.size() - 1) { c.pop_back(); return result; }
        c[selected] = std::move(c.back());
        c.pop_back();
        // Repair only this heap slot. No full heap rebuild or extra node copy.
        size_t i = selected;
        if (i && comp(c[(i - 1) / 2], c[i])) {
            while (i && comp(c[(i - 1) / 2], c[i])) {
                const size_t parent = (i - 1) / 2;
                std::swap(c[parent], c[i]); i = parent;
            }
        } else {
            while (i * 2 + 1 < c.size()) {
                size_t child = i * 2 + 1;
                if (child + 1 < c.size() && comp(c[child], c[child + 1])) ++child;
                if (!comp(c[i], c[child])) break;
                std::swap(c[i], c[child]); i = child;
            }
        }
        return result;
    }
    PersistentNode takeTop() {
        // pop_heap performs the same ordering operation as priority_queue::pop.
        // The selected node now belongs to the caller, so move its board/line
        // instead of deep-copying every card pile from the const top() view.
        std::pop_heap(c.begin(), c.end(), comp);
        PersistentNode result = std::move(c.back());
        c.pop_back();
        return result;
    }
};

struct RootPlan {
    Move first;
    std::vector<Move> line;
    int score = -10000000;
    int progress = 0;
    bool wins = false;
};

static bool betterRootPlan(bool wins, int score, size_t actions,
                            const RootPlan& existing) {
    if (wins != existing.wins) return wins;
    return score > existing.score ||
           (score == existing.score && actions < existing.line.size());
}

// Immutable inputs captured on the owner thread. The same planner can consume
// these on a worker without reading Unity, progress globals or mutable history.
struct SearchEnvironment {
    int baselineHidden = 0, baselineFoundation = 0, baselineDebt = 0;
    int recoveryPreparations = 0, movesWithoutProgress = 0, escalation = 0;
    int sliceNodes = 6400, sliceMs = 4, arenaNodes = 320000;
    bool recovery = false;
    bool stockMacros = true;
    bool balanceRootAlternatives = true;
    bool proveFoundationSuffix = true;
    std::unordered_set<std::string> failedFirstEdges, recentBoards, deadBoards;
    std::unordered_map<std::string, int> strategicFailures;
    std::vector<std::string> newDeadBoards;
};

struct PersistentExpertPlanner {
    std::string rootKey;
    int rootRecycles = 0;
    PersistentFrontier frontier;
    SearchDeck deck;
    SearchBoard rootBoard;
    // Root-scoped, independently replayed full continuation. Partial progress
    // can never overwrite it, even with a larger heuristic score.
    RootPlan winningCertificate;
    bool hasWinningCertificate = false;
    std::vector<SearchPathLink> paths;
    bool capacityReached = false;
    SearchEnvironment environment;
    std::unordered_map<SearchVisitKey, int, SearchVisitHash> bestPathCost;
    // Multiple root edges often converge after a Draw-3 rotation. Reuse the
    // complete structural evaluation for that exact canonical position.
    std::unordered_map<SearchPositionKey, int, SearchPositionHash> structuralScoreCache;
    std::unordered_map<SearchPositionKey, bool, SearchPositionHash> dependencyProofCache;
    // Only stock order/visibility is cached. Tableau/Foundation destinations
    // and charged-recycle costs are recalculated for each current position.
    StockOrbitCache stockOrbitCache;
    std::unordered_set<SearchPositionKey, SearchPositionHash> blockedFoundationSuffixes;
    std::unordered_map<uint32_t, int> rootCoverage;
    bool rootWarmup = false;
    // A first step rejected by certification invalidates every continuation
    // below that same edge for this root.  Keep this local to the current
    // planner so a later progress epoch can reconsider the physical move.
    std::unordered_set<std::string> uncertifiableFirstEdges;
    std::unordered_map<std::string, RootPlan> roots;
    // Keep one independently searched continuation behind the published best
    // line for each first edge.  A deeper certification failure must not erase
    // a legal first move when another already-proven continuation exists.
    std::unordered_map<std::string, RootPlan> rootBackups;
    // Risky Foundation edges remain searchable inside the frontier. They are
    // published only after their complete searched line reaches a proven
    // reveal, safe Foundation gain, debt reduction, or win.
    // Preparatory roots are populated only during strategic recovery.  They
    // preserve the normal solver's conservative behaviour while still giving
    // a stalled board a searched, non-random way to improve mobility, Waste
    // access, or dependency pressure before an irreversible move is possible.
    std::unordered_map<std::string, RootPlan> recoveryRoots;
    std::unordered_map<std::string, RootPlan> recoveryRootBackups;
    // Transposition is intentionally root-edge scoped. Two different first
    // decisions may converge on the same virtual state, but they must retain
    // independent evidence so a failed first branch cannot erase its sibling.
    std::unordered_map<std::string, int> rootHorizon;
    std::unordered_map<std::string, int> rootTerminalLeaves;
    int frames = 0;
    int expanded = 0;
    int maxDepth = 18;
    int rootHidden = 0;
    int rootFoundation = 0;
    int rootFoundationDebt = 0;
    int rootMobility = 0;
    int rootWasteAccess = 0;
    int rootDependencyPenalty = 0;
    int rootStructuralScore = -10000000;
    int rootKnownCards = 0;
    bool rootAutoCompleteLayout = false;
    bool endgame = false;
    bool perfectInformation = false;
    bool widenedProofGraph = false;
    bool solved = false;
    bool foundProgress = false;
};

static bool reconstructSearchLine(const PersistentExpertPlanner& planner,
                                  const PersistentNode& node,
                                  std::vector<Move>& line) {
    line.clear();
    line.reserve((size_t)node.depth);
    uint32_t cursor = node.path;
    for (int edges = 0; cursor != SOL_NO_PARENT; ++edges) {
        if (cursor >= planner.paths.size() || edges >= node.depth) return false;
        const auto& link = planner.paths[cursor];
        line.push_back(link.move);
        for (int i = 0; i < link.stockFlips; ++i)
            line.push_back({VirtualMoveKind::Talon, -1, -1, -1});
        cursor = link.parent;
    }
    if (line.size() != (size_t)node.depth) return false;
    std::reverse(line.begin(), line.end());
    return true;
}

// Delete-effect relaxation: targets and empty columns are never consumed;
// all stock cards are accessible; Foundation pullbacks remain available.
// A movable exposed suffix may disappear as a group, even if not ordered.
// These rules are STRICTLY more permissive than the actual forward game. If
// even this closure cannot put every card on Foundation, no real route can.
// Incomplete/invalid decks yield "not proved", never a pruning certificate.
static bool provenDependencyDeadEnd(const GameState& state) {
    if (!completeKnownDeck(state)) return false;
    std::array<bool, 52> accessible{}, removed{}, founded{};
    const auto index = [](const Card& card) {
        return knownCardSuit(card) * 13 + knownCardRank(card);
    };
    for (const auto& pile : state.foundations)
        for (const Card& card : pile)
            accessible[index(card)] = removed[index(card)] = founded[index(card)] = true;
    for (const auto& pile : {&state.talon, &state.waste})
        for (const Card& card : *pile) accessible[index(card)] = true;
    bool changed = true;
    while (changed) {
        changed = false;
        const auto set = [&](bool& bit) {
            if (!bit) { bit = true; changed = true; }
        };
        for (const auto& pile : state.tableau) {
            bool aboveRemovable = true;
            for (int i = (int)pile.size() - 1; i >= 0; --i) {
                const int id = index(pile[i]);
                if (aboveRemovable) set(accessible[id]);
                aboveRemovable = aboveRemovable && removed[id];
            }
        }
        const auto tableauDestination = [&](int id) {
            const int suit = id / 13, rank = id % 13;
            if (rank == 12) return true; // relaxed unlimited empty columns
            for (int other = 0; other < 4; ++other)
                if ((other & 1) != (suit & 1) && accessible[other * 13 + rank + 1])
                    return true;
            return false;
        };
        for (int id = 0; id < 52; ++id) {
            if (!accessible[id]) continue;
            if (id % 13 == 0 || founded[id - 1]) {
                set(founded[id]);
                set(removed[id]);
            }
            if (tableauDestination(id)) set(removed[id]);
        }
        // Do not require the bottom of a face-up sequence to be topmost:
        // dragging that entire suffix is legal in the real implementation.
        for (const auto& pile : state.tableau) {
            for (size_t start = 0; start < pile.size(); ++start) {
                const int id = index(pile[start]);
                if ((pile[start].faceUp() || accessible[id]) && tableauDestination(id))
                    for (size_t end = start; end < pile.size(); ++end)
                        set(removed[index(pile[end])]);
            }
        }
    }
    return std::find(founded.begin(), founded.end(), false) != founded.end();
}

struct ProgressMetrics {
    int foundation = 0;
    int foundationDebt = 0;
    int hidden = 0;
    int mobility = 0;
    int productiveMoves = 0;
    int emptyColumns = 0;
    int usefulWasteAccess = 0;
    int stockCycles = 0;
    int repeatedStates = 0;
    int planLength = 0;
    int structuralScore = 0;
    int knownCards = 0;
};

struct DecisionCheckpoint {
    uint64_t id = 0;
    int progressEpoch = 0;
    uint64_t createdAt = 0;
    int moveSerial = 0;
    GameState virtualRoot;
    ProgressMetrics metrics;
    std::string rootExactKey;
    std::string rootSolverKey;
    Move selected;
    std::string selectedMoveKey;
    std::string selectedResultKey;
    std::vector<Move> selectedLine;
    std::vector<std::string> alternatives;
    std::vector<std::string> pathStateKeys;
    int score = -10000000;
    bool failed = false;
};

struct StrategicProgressTracker {
    bool initialized = false;
    std::string exactKey;
    std::string physicalKey;
    ProgressMetrics baseline;
    ProgressMetrics current;
    int movesWithoutProgress = 0;
    int emergencyPreparations = 0;
    int recoveryPreparations = 0;
    int progressEpoch = 0;
    int moveSerial = 0;
    int recoveryAttemptsAtState = 0;
    uint64_t stateChangedAt = 0;
    bool stallPending = false;
    std::string stallReason;
    std::unordered_map<std::string, int> visitsSinceProgress;
};

static PersistentExpertPlanner g_persistentPlanner;
static std::unordered_set<std::string> g_deadStateCache;
static std::unordered_map<std::string, std::unordered_set<std::string>> g_failedRootMoves;
static std::unordered_map<std::string, std::unordered_set<std::string>> g_failedStrategicBranches;
static std::unordered_map<std::string, int> g_strategicStateFailures;
static std::deque<DecisionCheckpoint> g_decisionCheckpoints;
static StrategicProgressTracker g_progressTracker;
static uint64_t g_nextCheckpointId = 1;
static bool g_recoveryMode = false;
static int hiddenCardCount(const GameState& state);
struct DurableProgressEvidence {
    bool won = false;
    bool revealed = false;
    bool safeFoundation = false;
    bool repaidDebt = false;
    bool any() const {
        return won || revealed || safeFoundation || repaidDebt;
    }
};

// All physical guards and planner certificates must use the same high-water
// baseline. A Foundation card moved down and then put back is not fresh
// progress, even when the second move is safe relative to its local root.
static DurableProgressEvidence durableProgressFrom(
        const GameState& after, int baselineHidden,
        int baselineFoundation, int baselineDebt) {
    const int foundation = virtualFoundationCount(after);
    const int debt = foundationRiskDebt(after.foundations);
    return {foundation == 52,
            hiddenCardCount(after) < baselineHidden,
            foundation > baselineFoundation && debt <= baselineDebt,
            foundation >= baselineFoundation && debt < baselineDebt};
}
static uint64_t g_recoveryCheckpointId = 0;
static uint64_t g_lastRecoveryAt = 0;
static int g_recoveryGraceTransitions = 0;
// Once every checkpoint and the broader recovery frontier are exhausted for
// one exact physical state, park that state instead of re-entering recovery
// on every UI/raycaster tick.  Any real board change invalidates the latch.
static std::string g_exhaustedRecoveryKey;
static uint64_t g_exhaustedRecoveryAt = 0;
// Recovery depth is remembered per exact physical state.  A state first gets
// the normal expert search, then progressively wider bounded passes; it is
// never allowed to rebuild the same shallow frontier forever.
static std::unordered_map<std::string, int> g_recoveryEscalationByState;
// A relaxed emergency edge may be tried at most once from an exact physical
// state.  This preserves forward motion after the strategic planner has proved
// that no progress-certified root remains, without replacing a freeze with a
// Tableau or Draw-3 ping-pong loop.
static std::unordered_map<std::string, std::unordered_set<std::string>>
    g_emergencyEdgesTried;
// A repeated stock state may still expose a *new* legal exit after Draw-3.
// Admit that bridge at most once per exact root until genuine progress, rather
// than disabling the canonical loop guard for every Talon click.
static std::unordered_set<std::string> g_talonBridgeAttempted;
// If every emergency candidate was rejected for an unchanged exact state,
// remember that result. A physical move or newly learned hidden value changes
// the exact key and automatically re-enables emergency analysis.
static std::string g_emergencyNoMoveKey;
// Emergency look-ahead is evaluated in bounded batches on Unity's thread. Keep
// per-root coverage so a 20 ms slice examines the next candidates on the next
// frame instead of rechecking only the first eight and falsely latching no-move.
static std::string g_emergencyEvaluationRootKey;
static std::unordered_set<std::string> g_emergencyEvaluatedEdges;
static bool g_waitingForMoveResult = false;
static std::string g_pendingRootKey;
static std::string g_pendingExpectedKey;
static std::string g_pendingPhysicalRootKey;
static std::string g_pendingPhysicalExpectedKey;
static std::string g_pendingMoveKey;
static uint64_t g_pendingMoveAt = 0;
static int g_pendingRecycleCountBefore = 0;
static int g_pendingStagnantRecycleCountBefore = 0;
// A submitted action stays in flight until one physical candidate remains
// stable. These fields separate first observation from verified settlement.
static std::string g_pendingObservedPhysicalKey;
static uint64_t g_pendingObservedSinceMs = 0;
static bool g_pendingSoftTimeoutLogged = false;
// Jawaker's own optimistic queue is diagnostic evidence only; it is not used
// as a success verdict because SyncFromGameStore may remove/reconcile entries.
static int64_t g_pendingSequence = -1;
static int64_t g_pendingSequenceBefore = -1;
static int g_pendingCountAtSubmit = 0;
static bool g_pendingWasQueued = false;
// Server-mode pipelining. The pending flight above is always the oldest
// unverified action; g_chainedSteps are committed-plan steps already sent
// behind it, oldest first. Each is promoted to the pending flight in turn.
struct ChainedStep {
    GameState expected;                // board after this step
    std::string expectedKey;           // virtualPositionKey(expected)
    std::string physicalExpectedKey;   // physicalPositionKey(expected)
    std::string moveKey;
    uint64_t sentAt = 0;
    int recycleCountBefore = 0;
    int stagnantRecycleCountBefore = 0;
    uint64_t checkpointId = 0;
    bool flipsTableau = false;         // turns a face-down Tableau card up
};
static std::deque<ChainedStep> g_chainedSteps;
static GameState g_pendingExpectedState;   // board after the pending flight
static bool g_pendingChainable = false;    // pending flight came from a plan
// Hidden Tableau values may be inferred rather than read; nothing is queued
// behind a flip until the server shows the real card.
static bool g_pendingFlipsTableau = false;
static uint64_t g_pipelineDrainUntilMs = 0;
static int g_pendingFoundationRescueCardId = -1;
static bool g_pendingCountsEmergencyPreparation = false;
static bool g_pendingCountsRecoveryPreparation = false;
static uint64_t g_pendingCheckpointId = 0;
static std::vector<Move> g_committedPlan;
static int g_committedPlanScore = -10000000;
static bool g_committedPlanAllowsRecoveryPreparation = false;
static bool g_committedPlanIsVerifiedWin = false;
static std::string g_committedWinExpectedKey;
static int g_committedWinExpectedRecycles = 0;
static std::string g_foundationFinishAttemptedKey;
struct FoundationFinishProbe {
    std::string rootKey;
    GameState cursor;
    std::vector<Move> line;
    SearchDeck deck;
    SearchBoard board;
    bool compactReady = false;
    std::unordered_set<SearchPositionKey, SearchPositionHash> visited;
    int steps = 0;
};
static FoundationFinishProbe g_foundationFinishProbe;

static void armPendingFlightObservation() {
    g_pendingObservedPhysicalKey.clear();
    g_pendingObservedSinceMs = 0;
    g_pendingSoftTimeoutLogged = false;
    g_pendingSequence = g_lastSubmittedSequence;
    g_pendingSequenceBefore = g_lastSubmittedSequenceBefore;
    g_pendingCountAtSubmit = g_lastSubmittedPendingCount;
    g_pendingWasQueued = g_lastSubmittedWasQueued;
    g_pendingFoundationRescueCardId =
        g_lastSubmittedFoundationRescueCardId;
    g_pendingCountsEmergencyPreparation = false;
    g_pendingCountsRecoveryPreparation = false;
    g_pendingCheckpointId = 0;
    // Only the two planner dispatch sites opt back in, after arming.
    g_pendingChainable = false;
}

static void clearPlannerFrontier() {
    g_persistentPlanner = PersistentExpertPlanner{};
}

static void cancelAnticipatorySearch();

static void resetExpertPlanner(bool releaseCapacity) {
    cancelAnticipatorySearch();
    clearPlannerFrontier();
    g_deadStateCache.clear();
    g_failedRootMoves.clear();
    g_failedStrategicBranches.clear();
    g_strategicStateFailures.clear();
    g_decisionCheckpoints.clear();
    g_progressTracker = StrategicProgressTracker{};
    g_nextCheckpointId = 1;
    g_recoveryMode = false;
    g_recoveryCheckpointId = 0;
    g_lastRecoveryAt = 0;
    g_recoveryGraceTransitions = 0;
    g_exhaustedRecoveryKey.clear();
    g_exhaustedRecoveryAt = 0;
    g_recoveryEscalationByState.clear();
    g_emergencyEdgesTried.clear();
    g_talonBridgeAttempted.clear();
    g_emergencyNoMoveKey.clear();
    g_emergencyEvaluationRootKey.clear();
    g_emergencyEvaluatedEdges.clear();
    g_waitingForMoveResult = false;
    g_pendingRootKey.clear();
    g_pendingExpectedKey.clear();
    g_pendingPhysicalRootKey.clear();
    g_pendingPhysicalExpectedKey.clear();
    g_pendingMoveKey.clear();
    g_pendingMoveAt = 0;
    g_pendingRecycleCountBefore = 0;
    g_pendingStagnantRecycleCountBefore = 0;
    g_pendingObservedPhysicalKey.clear();
    g_pendingObservedSinceMs = 0;
    g_pendingSoftTimeoutLogged = false;
    g_pendingSequence = -1;
    g_pendingSequenceBefore = -1;
    g_pendingCountAtSubmit = 0;
    g_pendingWasQueued = false;
    g_pendingFoundationRescueCardId = -1;
    g_pendingCountsEmergencyPreparation = false;
    g_pendingCountsRecoveryPreparation = false;
    g_pendingCheckpointId = 0;
    g_chainedSteps.clear();
    g_pendingExpectedState = GameState{};
    g_pendingChainable = false;
    g_pendingFlipsTableau = false;
    g_pipelineDrainUntilMs = 0;
    g_committedPlan.clear();
    g_committedPlanScore = -10000000;
    g_committedPlanAllowsRecoveryPreparation = false;
    g_committedPlanIsVerifiedWin = false;
    g_committedWinExpectedKey.clear();
    g_committedWinExpectedRecycles = 0;
    g_foundationFinishAttemptedKey.clear();
    g_foundationFinishProbe = FoundationFinishProbe{};
    g_lastDispatchOutcome = DispatchOutcome::Rejected;
    g_dispatchDeferredThisTick = false;
    g_dispatchRetryAtMs = 0;
    g_dispatchRetryPhysicalKey.clear();
    g_cardViewMissSinceMs = 0;
    g_lastCardViewRefreshMs = 0;
    g_cardViewMissId = -1;
    resetLastSubmissionObservation();
    if (releaseCapacity) {
        // clear() deliberately retains buckets/capacity. At an actual deal
        // boundary none of these records may be reused, so release their
        // backing allocations before asking bionic/Scudo to purge pages.
        decltype(g_deadStateCache)().swap(g_deadStateCache);
        decltype(g_failedRootMoves)().swap(g_failedRootMoves);
        decltype(g_failedStrategicBranches)().swap(g_failedStrategicBranches);
        decltype(g_strategicStateFailures)().swap(g_strategicStateFailures);
        decltype(g_decisionCheckpoints)().swap(g_decisionCheckpoints);
        decltype(g_recoveryEscalationByState)().swap(g_recoveryEscalationByState);
        decltype(g_emergencyEdgesTried)().swap(g_emergencyEdgesTried);
        decltype(g_talonBridgeAttempted)().swap(g_talonBridgeAttempted);
        decltype(g_emergencyEvaluatedEdges)().swap(g_emergencyEvaluatedEdges);
        decltype(g_committedPlan)().swap(g_committedPlan);
        decltype(g_progressTracker.visitsSinceProgress)().swap(
            g_progressTracker.visitsSinceProgress);
        decltype(g_foundationFinishProbe.line)().swap(g_foundationFinishProbe.line);
        decltype(g_foundationFinishProbe.visited)().swap(
            g_foundationFinishProbe.visited);
    }
}

static int hiddenCardCount(const GameState& state) {
    int hidden = 0;
    for (const auto& column : state.tableau)
        for (const Card& card : column)
            if (!card.revealedForPlanning()) ++hidden;
    return hidden;
}

// Prove a complete endgame using only monotone Foundation advances and exact
// native Draw-N transitions. A legal Foundation advance cannot remove another
// Foundation move, so no branching is needed here. The runtime proof resumes
// across bounded Unity frames instead of declaring a solvable line impossible
// merely because one 5 ms slice expired.
enum class FoundationFinishResult : uint8_t { Pending, Won, Blocked };

static FoundationFinishResult advanceFoundationOnlyFinishPlan(
        FoundationFinishProbe& probe, int stepBudget, uint64_t deadlineMs) {
    if (!probe.compactReady) {
        if (!probe.deck.initialize(probe.cursor) ||
            !packSearchBoard(probe.cursor, probe.deck, probe.board))
            return FoundationFinishResult::Blocked;
        reserveSearchScratch(probe.cursor);
        probe.compactReady = true;
    }
    // Public probe snapshots remain compatible at every yield/return. Inside
    // the proof loop only the fixed-size board changes, with no card-vector
    // copy, full string key or repeated full-board scan per Foundation move.
    struct SyncCursor {
        FoundationFinishProbe& probe;
        ~SyncCursor() { unpackSearchBoard(probe.board, probe.deck, probe.cursor); }
    } sync{probe};
    const auto topCard = [&](int pile) {
        unsigned end = 0;
        for (int i = 0; i <= pile; ++i) end += probe.board.sizes[i];
        const uint8_t token = probe.board.cards[end - 1];
        const Card& base = probe.deck.cards[token & 0x3f];
        Card card;
        card.id = base.id;
        card.knownValue = base.knownValue;
        card.value = token & 0x40 ? base.knownValue : -1;
        card.virtuallyRevealed = (token & 0x80) != 0;
        return card;
    };
    std::array<int, 4> targetBySuit{-1, -1, -1, -1};
    std::array<int, 4> nextRank{0, 0, 0, 0};
    int emptyTarget = -1;
    for (int dst = 0; dst < 4; ++dst) {
        if (!probe.board.sizes[7 + dst]) { if (emptyTarget < 0) emptyTarget = dst; continue; }
        const Card top = topCard(7 + dst);
        const int suit = knownCardSuit(top);
        if (suit < 0 || suit >= 4) return FoundationFinishResult::Blocked;
        targetBySuit[suit] = dst;
        nextRank[suit] = knownCardRank(top) + 1;
    }
    int foundationCount = 0;
    for (int pile = 7; pile < 11; ++pile) foundationCount += probe.board.sizes[pile];
    for (int batch = 0; batch < stepBudget; ++batch) {
        if (foundationCount == 52)
            return FoundationFinishResult::Won;
        if (probe.steps >= 4096) return FoundationFinishResult::Blocked;
        if (deadlineMs && nowMs() > deadlineMs)
            return FoundationFinishResult::Pending;
        if (!probe.visited.insert(searchPositionKey(probe.board, probe.deck, true)).second)
            return FoundationFinishResult::Blocked;

        Move chosen;
        bool found = false;
        const auto foundationTarget = [&](const Card& card) {
            if (!card.faceUp()) return -1;
            const int suit = cardSuit(card), rank = cardRank(card);
            if (suit < 0 || rank != nextRank[suit]) return -1;
            return targetBySuit[suit] >= 0 ? targetBySuit[suit] : emptyTarget;
        };
        if (probe.board.sizes[12]) {
            const int dst = foundationTarget(topCard(12));
            if (dst >= 0) {
                chosen = {VirtualMoveKind::Foundation, -1, dst, -1};
                found = true;
            }
        }
        for (int src = 0; src < 7 && !found; ++src) {
            if (!probe.board.sizes[src]) continue;
            const int dst = foundationTarget(topCard(src));
            if (dst >= 0) {
                chosen = {VirtualMoveKind::Foundation, src, dst, -1};
                found = true;
            }
        }
        if (!found) {
            if (!probe.board.sizes[11] && !probe.board.sizes[12])
                return FoundationFinishResult::Blocked;
            chosen = {VirtualMoveKind::Talon, -1, -1, -1};
        }
        if (!applySearchMove(probe.board, probe.deck, chosen))
            return FoundationFinishResult::Blocked;
        probe.line.push_back(chosen);
        ++probe.steps;
        if (chosen.kind == VirtualMoveKind::Foundation) {
            const Card card = topCard(7 + chosen.to);
            const int suit = cardSuit(card);
            targetBySuit[suit] = chosen.to;
            nextRank[suit] = cardRank(card) + 1;
            ++foundationCount;
            emptyTarget = -1;
            for (int dst = 0; dst < 4; ++dst)
                if (!probe.board.sizes[7 + dst]) { emptyTarget = dst; break; }
            // Monotone gain cannot return to an earlier Foundation position.
            // Only the current stock-only cycle needs duplicate tracking.
            probe.visited.clear();
        }
        if (foundationCount == 52)
            return FoundationFinishResult::Won;
    }
    return FoundationFinishResult::Pending;
}

static bool findFoundationOnlyFinishPlan(const GameState& root,
                                         std::vector<Move>& plan) {
    plan.clear();
    if (!completeKnownDeck(root) || hasOpaqueCards(root) ||
        hiddenCardCount(root) > SOL_VICTORY_ENDGAME_HIDDEN) return false;
    FoundationFinishProbe probe;
    probe.cursor = root;
    probe.line.reserve(96);
    probe.visited.reserve(192);
    const FoundationFinishResult result =
        advanceFoundationOnlyFinishPlan(probe, 4096, 0);
    if (result != FoundationFinishResult::Won) return false;
    plan = std::move(probe.line);
    return true;
}

static bool verifiedWinningLine(const GameState& root,
                                const std::vector<Move>& line) {
    if (line.empty() || !completeKnownDeck(root) || hasOpaqueCards(root))
        return false;
    SearchDeck deck;
    SearchBoard board;
    if (!deck.initialize(root) || !packSearchBoard(root, deck, board)) return false;
    std::unordered_set<SearchPositionKey, SearchPositionHash> visited;
    visited.reserve(line.size() + 1);
    visited.insert(searchPositionKey(board, deck, true));
    for (size_t i = 0; i < line.size(); ++i) {
        const Move& move = line[i];
        if (!applySearchMove(board, deck, move)) return false;
        int count = 0;
        for (int pile = 7; pile < 11; ++pile) count += board.sizes[pile];
        if (count == 52)
            return i + 1 == line.size();
        if (!visited.insert(searchPositionKey(board, deck, true)).second) return false;
    }
    return false;
}

static bool retainPlannerWinningCertificate(PersistentExpertPlanner& planner,
                                             const RootPlan& candidate) {
    if (!candidate.wins || candidate.line.empty() ||
        searchMoveId(candidate.first) != searchMoveId(candidate.line.front())) return false;
    GameState root;
    if (!unpackSearchBoard(planner.rootBoard, planner.deck, root) ||
        !verifiedWinningLine(root, candidate.line)) return false;
    if (!planner.hasWinningCertificate || betterRootPlan(true, candidate.score,
            candidate.line.size(), planner.winningCertificate))
        planner.winningCertificate = candidate;
    planner.hasWinningCertificate = planner.solved = planner.foundProgress = true;
    return true;
}

static void trackCommittedWinState(const GameState& expected) {
    if (g_committedPlanIsVerifiedWin && !g_committedPlan.empty()) {
        g_committedWinExpectedKey = virtualPositionKey(expected);
        g_committedWinExpectedRecycles = expected.recycles;
    } else {
        g_committedWinExpectedKey.clear();
    }
}

static bool committedWinMatches(const GameState& observed) {
    return g_committedPlanIsVerifiedWin && !g_committedPlan.empty() &&
        observed.recycles == g_committedWinExpectedRecycles &&
        g_committedWinExpectedKey == virtualPositionKey(observed);
}

static bool hasUsefulDestination(const GameState& state, const Card& original) {
    if (!original.known()) return false;
    Card card = original;
    if (!card.faceUp()) card.value = card.knownValue;
    for (int dst = 0; dst < 4; ++dst) {
        const Card* top = state.foundations[dst].empty() ? nullptr :
                          &state.foundations[dst].back();
        if (canDropOnFoundation(card, top)) return true;
    }
    for (int dst = 0; dst < 7; ++dst) {
        const Card* top = state.tableau[dst].empty() ? nullptr :
                          &state.tableau[dst].back();
        if (canDropOnTableau(card, top)) return true;
    }
    return false;
}

static int usefulWasteAccessScore(const GameState& state) {
    int access = 0;
    if (!state.waste.empty() && hasUsefulDestination(state, state.waste.back()))
        access += 4; // immediately executable exposed Waste card

    // Evaluate the exact Draw-N exposure points, not every stock card.  With
    // the native ordering confirmed in OnTalonClicked, the last card of each
    // front draw batch becomes Waste.back().  This metric therefore measures
    // real future accessibility rather than merely noticing a buried card.
    const int draw = std::clamp(state.drawCount, 1, 3);
    for (int offset = 0; offset < (int)state.talon.size(); offset += draw) {
        const int exposed = std::min(offset + draw - 1, (int)state.talon.size() - 1);
        if (hasUsefulDestination(state, state.talon[exposed])) ++access;
    }
    return access;
}

static int productiveMoveCount(const GameState& state) {
    std::vector<Move> moves;
    generateVirtualMoves(state, moves);
    const int hiddenBefore = hiddenCardCount(state);
    const int mobilityBefore = virtualMobility(state);
    int productive = 0;
    for (const Move& move : moves) {
        if (move.kind == VirtualMoveKind::Talon ||
            move.kind == VirtualMoveKind::FoundationToTableau) continue;
        GameState after = state;
        if (!applyVirtualMove(after, move)) continue;
        const bool reveals = hiddenCardCount(after) < hiddenBefore;
        const bool safeFoundation = move.kind == VirtualMoveKind::Foundation &&
                                    foundationRiskDebt(after.foundations) <=
                                    foundationRiskDebt(state.foundations);
        const bool freesWaste = move.from < 0;
        const bool improvesMobility = virtualMobility(after) > mobilityBefore;
        if (reveals || safeFoundation || freesWaste || improvesMobility) ++productive;
    }
    return productive;
}

static ProgressMetrics collectProgressMetrics(const GameState& state, int planLength) {
    ProgressMetrics metrics;
    metrics.foundation = virtualFoundationCount(state);
    metrics.foundationDebt = foundationRiskDebt(state.foundations);
    metrics.hidden = hiddenCardCount(state);
    metrics.mobility = virtualMobility(state);
    metrics.productiveMoves = productiveMoveCount(state);
    metrics.usefulWasteAccess = usefulWasteAccessScore(state);
    metrics.stockCycles = state.stagnantRecycles;
    metrics.planLength = planLength;
    metrics.structuralScore = virtualStructuralScore(state);
    for (const auto& column : state.tableau) {
        if (column.empty()) ++metrics.emptyColumns;
        for (const Card& card : column) if (card.known()) ++metrics.knownCards;
    }
    for (const auto& pile : state.foundations)
        for (const Card& card : pile) if (card.known()) ++metrics.knownCards;
    for (const Card& card : state.talon) if (card.known()) ++metrics.knownCards;
    for (const Card& card : state.waste) if (card.known()) ++metrics.knownCards;
    return metrics;
}

static void pruneOldCheckpoints() {
    while (!g_decisionCheckpoints.empty() &&
           (g_decisionCheckpoints.size() > 16 ||
            g_decisionCheckpoints.front().progressEpoch + 1 <
                g_progressTracker.progressEpoch))
        g_decisionCheckpoints.pop_front();
}

static void flagStrategicStall(const char* reason) {
    if (g_progressTracker.stallPending) return;
    g_progressTracker.stallPending = true;
    g_progressTracker.stallReason = reason ? reason : "unknown";
    const ProgressMetrics& base = g_progressTracker.baseline;
    const ProgressMetrics& now = g_progressTracker.current;
    SOL_LOG("[STALL] reason=%s movesWithoutProgress=%d foundationDelta=%d hiddenDelta=%d mobilityDelta=%d productive=%d empty=%d wasteAccess=%d stockCycles=%d repeated=%d planLength=%d known=%d",
            g_progressTracker.stallReason.c_str(),
            g_progressTracker.movesWithoutProgress,
            now.foundation - base.foundation, base.hidden - now.hidden,
            now.mobility - base.mobility, now.productiveMoves,
            now.emptyColumns, now.usefulWasteAccess, now.stockCycles,
            now.repeatedStates, now.planLength, now.knownCards);
}

static void observeStrategicProgress(const GameState& state, uint64_t now) {
    const std::string exactKey = virtualPositionKey(state);
    const std::string physicalKey = physicalPositionKey(state);
    if (!g_exhaustedRecoveryKey.empty() && g_exhaustedRecoveryKey != exactKey) {
        g_exhaustedRecoveryKey.clear();
        g_exhaustedRecoveryAt = 0;
    }
    if (g_progressTracker.initialized &&
        g_progressTracker.physicalKey == physicalKey) {
        // Newly learned hidden values change the information-rich planner
        // key, not the physical board. Re-evaluate the improved information
        // without inventing a move, a repeat visit or a no-progress strike.
        if (g_progressTracker.exactKey != exactKey) {
            ProgressMetrics refreshed = collectProgressMetrics(
                state, (int)g_committedPlan.size());
            refreshed.repeatedStates = g_progressTracker.current.repeatedStates;
            g_progressTracker.current = refreshed;
            g_progressTracker.exactKey = exactKey;
            g_progressTracker.stateChangedAt = now;
            g_progressTracker.stallPending = false;
            g_progressTracker.stallReason.clear();
        } else {
            // The UI and raycaster hooks may both tick several times per frame.
            g_progressTracker.current.planLength = (int)g_committedPlan.size();
            g_progressTracker.current.stockCycles = state.stagnantRecycles;
        }
        return;
    }
    ProgressMetrics metrics = collectProgressMetrics(state, (int)g_committedPlan.size());
    if (!g_progressTracker.initialized) {
        g_progressTracker.initialized = true;
        g_progressTracker.exactKey = exactKey;
        g_progressTracker.physicalKey = physicalKey;
        g_progressTracker.baseline = metrics;
        g_progressTracker.current = metrics;
        g_progressTracker.stateChangedAt = now;
        metrics.repeatedStates = ++g_progressTracker.visitsSinceProgress[solverPositionKey(state)];
        g_progressTracker.current.repeatedStates = metrics.repeatedStates;
        return;
    }
    g_progressTracker.exactKey = exactKey;
    g_progressTracker.physicalKey = physicalKey;
    g_progressTracker.stateChangedAt = now;
    g_progressTracker.recoveryAttemptsAtState = 0;
    ++g_progressTracker.moveSerial;
    // Compare against the last proven high-water baseline. Comparing only to
    // the previous frame incorrectly treated Foundation -> Tableau ->
    // Foundation as fresh progress and reset all loop/recovery evidence.
    // A raw Foundation-count increase is not progress when it creates new
    // tactical debt.  Treating it as progress used to clear the loop evidence
    // exactly when one suit was being rushed ahead of the opposite colours.
    const bool safeFoundationProgress =
        metrics.foundation > g_progressTracker.baseline.foundation &&
        metrics.foundationDebt <= g_progressTracker.baseline.foundationDebt;
    const bool debtRepaidWithoutFoundationRetreat =
        metrics.foundation >= g_progressTracker.baseline.foundation &&
        metrics.foundationDebt < g_progressTracker.baseline.foundationDebt;
    const bool irreversibleProgress =
        metrics.foundation == 52 || safeFoundationProgress ||
        debtRepaidWithoutFoundationRetreat ||
        metrics.hidden < g_progressTracker.baseline.hidden;
    if (irreversibleProgress) {
        ++g_progressTracker.progressEpoch;
        g_progressTracker.movesWithoutProgress = 0;
        g_progressTracker.emergencyPreparations = 0;
        g_progressTracker.recoveryPreparations = 0;
        g_progressTracker.baseline = metrics;
        g_progressTracker.visitsSinceProgress.clear();
        g_talonBridgeAttempted.clear();
        g_progressTracker.stallPending = false;
        g_progressTracker.stallReason.clear();
        g_recoveryMode = false;
        g_recoveryCheckpointId = 0;
        g_recoveryGraceTransitions = 0;
        g_recoveryEscalationByState.clear();
        pruneOldCheckpoints();
    } else {
        ++g_progressTracker.movesWithoutProgress;
        if (g_recoveryGraceTransitions > 0) --g_recoveryGraceTransitions;
    }

    metrics.repeatedStates = ++g_progressTracker.visitsSinceProgress[solverPositionKey(state)];
    g_progressTracker.current = metrics;
    if (irreversibleProgress || g_recoveryGraceTransitions > 0) return;

    const int stockCards = (int)state.talon.size() + (int)state.waste.size();
    const int draw = std::clamp(state.drawCount, 1, 3);
    const int fullPassSteps = std::max(1, (stockCards + draw - 1) / draw + 1);
    int threshold = std::max(10, fullPassSteps + 5);
    if (metrics.hidden <= 7) threshold += 4;
    if (metrics.productiveMoves >= 4) threshold += 2;
    threshold = std::min(threshold, 32);

    const int mobilityDelta = metrics.mobility - g_progressTracker.baseline.mobility;
    const int productiveDelta = metrics.productiveMoves -
                                g_progressTracker.baseline.productiveMoves;
    if (metrics.repeatedStates >= 2 &&
        g_progressTracker.movesWithoutProgress >= std::max(6, fullPassSteps)) {
        flagStrategicStall("repeated-tactical-state");
    } else if (metrics.stockCycles >= 2 && metrics.usefulWasteAccess == 0 &&
               g_progressTracker.movesWithoutProgress >= fullPassSteps * 2) {
        flagStrategicStall("draw3-recycle-without-progress");
    } else if (g_progressTracker.movesWithoutProgress >= threshold &&
               mobilityDelta < 0 && productiveDelta <= 0 &&
               metrics.productiveMoves <= 2) {
        flagStrategicStall("mobility-collapse");
    } else if (g_progressTracker.movesWithoutProgress >= threshold * 2 &&
               (metrics.repeatedStates >= 2 || metrics.stockCycles > 0 ||
                metrics.productiveMoves == 0)) {
        flagStrategicStall("bounded-no-progress");
    }
}

static bool physicalCandidateAllowed(const GameState& state, const Move& move, uint64_t now);
static int immediateMoveTier(const GameState& state, const Move& move);

static SearchEnvironment captureSearchEnvironment(const GameState& state,
                                                   const std::string& key) {
    SearchEnvironment environment;
    const auto& epoch = g_progressTracker.baseline;
    environment.baselineHidden = g_progressTracker.initialized ? epoch.hidden : hiddenCardCount(state);
    environment.baselineFoundation = g_progressTracker.initialized ? epoch.foundation : virtualFoundationCount(state);
    environment.baselineDebt = g_progressTracker.initialized ? epoch.foundationDebt : foundationRiskDebt(state.foundations);
    environment.recovery = g_recoveryMode;
    environment.recoveryPreparations = g_progressTracker.recoveryPreparations;
    environment.movesWithoutProgress = g_progressTracker.movesWithoutProgress;
    const auto escalation = g_recoveryEscalationByState.find(key);
    environment.escalation = escalation == g_recoveryEscalationByState.end() ? 0 : escalation->second;
    environment.sliceNodes = g_searchSliceNodes.load(std::memory_order_relaxed);
    environment.sliceMs = g_searchSliceMs.load(std::memory_order_relaxed);
    environment.arenaNodes = g_searchArenaNodes.load(std::memory_order_relaxed);
    const uint64_t now = nowMs();
    for (const auto& entry : g_solverStateHistory)
        if (now >= entry.at && now - entry.at <= 180000) environment.recentBoards.insert(entry.tableau);
    environment.deadBoards = g_deadStateCache;
    environment.strategicFailures = g_strategicStateFailures;
    const auto appendFailures = [&](const auto& failures) {
        const auto it = failures.find(key);
        if (it != failures.end())
            environment.failedFirstEdges.insert(it->second.begin(), it->second.end());
    };
    appendFailures(g_failedRootMoves);
    appendFailures(g_failedStrategicBranches);
    return environment;
}

static bool initializePlannerRoot(PersistentExpertPlanner& planner,
                                  const GameState& state, const std::string& key,
                                  SearchEnvironment environment) {
    planner = PersistentExpertPlanner{};
    planner.environment = std::move(environment);
    if (!planner.deck.initialize(state)) return false;
    planner.paths.reserve(32768);
    planner.rootKey = key;
    planner.rootRecycles = state.recycles;
    planner.rootHidden = hiddenCardCount(state);
    planner.rootFoundation = virtualFoundationCount(state);
    planner.rootFoundationDebt = foundationRiskDebt(state.foundations);
    planner.rootMobility = virtualMobility(state);
    planner.rootWasteAccess = usefulWasteAccessScore(state);
    planner.rootDependencyPenalty = dependencyPenalty(state);
    planner.rootStructuralScore = virtualStructuralScore(state);
    planner.structuralScoreCache.reserve(8192);
    planner.rootKnownCards = knownCardCount(state);
    planner.rootAutoCompleteLayout =
        provenAutoCompleteLayout(state);
    planner.perfectInformation = completeKnownDeck(state);
    planner.endgame = planner.rootHidden <= SOL_VICTORY_ENDGAME_HIDDEN ||
                      planner.rootFoundation >= SOL_VICTORY_ENDGAME_FOUNDATION;
    if (planner.perfectInformation) {
        // Perfect information makes long-horizon search deterministic rather
        // than speculative. Depth remains node/time bounded across frames.
        planner.maxDepth = planner.endgame ? 288 :
            (planner.rootHidden <= 18 ? 224 : 160);
    } else {
        // Preserve the proven conservative bounds while IL2CPP mapping is
        // incomplete; an unknown card must never be guessed into a plan.
        planner.maxDepth = planner.endgame ? 112 :
            (planner.rootHidden <= 14 ? 64 : 48);
    }
    if (planner.environment.recovery) {
        const int recoveryBoost = 16 * (1 + planner.environment.escalation);
        planner.maxDepth = std::min(320, planner.maxDepth + recoveryBoost);
    }
    PersistentNode root;
    if (!packSearchBoard(state, planner.deck, root.board)) return false;
    planner.rootBoard = root.board;
    root.hiddenCount = planner.rootHidden;
    root.foundationCount = planner.rootFoundation;
    root.autoCompleteProven = planner.rootAutoCompleteLayout;
    root.structuralScore = planner.rootStructuralScore;
    root.priority = planner.rootStructuralScore;
    planner.frontier.push(std::move(root));
    planner.bestPathCost.reserve(32768);
    planner.bestPathCost[{searchPositionKey(root.board, planner.deck), UINT32_MAX, false}] = 0;
    return true;
}

static void initializePersistentRoot(const GameState& state, const std::string& key) {
    initializePlannerRoot(g_persistentPlanner, state, key, captureSearchEnvironment(state, key));
    SOL_LOG("expert planner root endgame=%d recovery=%d depth=%d known=%d/52 perfect=%d escalation=%d deadCache=%d",
            g_persistentPlanner.endgame ? 1 : 0, g_recoveryMode ? 1 : 0,
            g_persistentPlanner.maxDepth, g_persistentPlanner.rootKnownCards,
            g_persistentPlanner.perfectInformation ? 1 : 0, g_persistentPlanner.environment.escalation,
            (int)g_deadStateCache.size());
}

static bool rootMoveFailed(const std::string& rootKey, const Move& move) {
    const std::string key = moveKey(move);
    if (g_persistentPlanner.rootKey == rootKey &&
        g_persistentPlanner.uncertifiableFirstEdges.find(key) !=
            g_persistentPlanner.uncertifiableFirstEdges.end()) return true;
    const auto physical = g_failedRootMoves.find(rootKey);
    if (physical != g_failedRootMoves.end() &&
        physical->second.find(key) != physical->second.end()) return true;
    const auto strategic = g_failedStrategicBranches.find(rootKey);
    return strategic != g_failedStrategicBranches.end() &&
           strategic->second.find(key) != strategic->second.end();
}

static int strategicStatePenalty(const std::string& key) {
    const auto it = g_strategicStateFailures.find(key);
    if (it == g_strategicStateFailures.end()) return 0;
    // Strategic failures are evidence, not a mathematical proof.  Penalise a
    // repeatedly bad structure heavily during recovery but never hard-delete
    // it; a later changed Waste/Foundation context may make it necessary.
    const int unit = g_recoveryMode ? 280000 : 65000;
    return std::min(1400000, it->second * unit);
}

static void recordRootPlan(PersistentExpertPlanner& planner,
                           const PersistentNode& node, const GameState& state) {
    if (node.path == SOL_NO_PARENT) return;
    const auto& environment = planner.environment;
    const int hiddenGain = planner.rootHidden - node.hiddenCount;
    const int foundationGain = node.foundationCount -
                                planner.rootFoundation;
    const int debtAfter = foundationRiskDebt(state.foundations);
    const int debtIncrease = std::max(0, debtAfter - planner.rootFoundationDebt);
    const int debtReduction = std::max(0, planner.rootFoundationDebt - debtAfter);
    const bool won = node.foundationCount == 52;
    const bool autoCompleteLayout =
        !planner.rootAutoCompleteLayout &&
        node.autoCompleteProven;
    // A plan becomes executable only after its branch proves irreversible
    // progress.  Talon-only rotations and cosmetic Tableau shuffles can still
    // be explored as preparation, but can never be selected on their own.
    // Merely dumping a waste card onto Tableau is deliberately not progress:
    // that old shortcut was the main source of mid-game self-congestion.
    // Likewise, a risky foundation push is executable only if this same line
    // also uncovers a hidden card or repays the foundation debt.
    const int baselineHidden = environment.baselineHidden;
    const int baselineFoundation = environment.baselineFoundation;
    const int baselineDebt = environment.baselineDebt;
    const DurableProgressEvidence durable{
        won, node.hiddenCount < baselineHidden,
        node.foundationCount > baselineFoundation && debtAfter <= baselineDebt,
        node.foundationCount >= baselineFoundation && debtAfter < baselineDebt};
    const bool provenProgress = durable.any() || autoCompleteLayout;
    // The frontier calls this for every child. Ordinary nodes without a
    // reveal/approved Foundation gain cannot be published, so do not rescan
    // all Tableau, Waste and dependency chains for them. Those costly metrics
    // are needed only to certify a recovery preparation.
    if (!provenProgress &&
        (!environment.recovery || debtIncrease != 0 || node.depth < 2 ||
         environment.recoveryPreparations >=
             SOL_MAX_RECOVERY_PREPARATIONS)) return;
    int mobilityGain = 0;
    int wasteAccessGain = 0;
    int dependencyGain = 0;
    int structuralGain = 0;
    bool recoveryPreparation = false;
    if (!provenProgress) {
        mobilityGain = virtualMobility(state) - planner.rootMobility;
        wasteAccessGain = usefulWasteAccessScore(state) -
                          planner.rootWasteAccess;
        dependencyGain = planner.rootDependencyPenalty -
                         dependencyPenalty(state);
        structuralGain = node.structuralScore -
                         planner.rootStructuralScore;
        recoveryPreparation = mobilityGain > 0 || wasteAccessGain > 0 ||
                              dependencyGain >= 120 || structuralGain >= 900;
    }
    // A risky Foundation move remains in the search graph, so a deeper line
    // can still prove that it reveals a card or repays the debt.  It must not
    // be published as an executable root merely because FoundationCount rose.
    if (!provenProgress && !recoveryPreparation) return;
    int progress = (won ? 2000000 : 0) +
                   (autoCompleteLayout ? 800000 : 0) + hiddenGain * 100000 +
                   foundationGain * 4200 + debtReduction * 12000 -
                   debtIncrease * 18000 - node.depth * 24;
    if (recoveryPreparation)
        progress += mobilityGain * 5200 + wasteAccessGain * 2800 +
                    dependencyGain * 22 + std::max(0, structuralGain) / 3;
    const int planScore = node.priority + progress;
    const std::string firstKey = moveKey(node.first);
    auto& destination = provenProgress ? planner.roots : planner.recoveryRoots;
    auto& backups = provenProgress ? planner.rootBackups : planner.recoveryRootBackups;
    const auto sameMove = [](const Move& a, const Move& b) {
        return a.kind == b.kind && a.from == b.from && a.to == b.to &&
               a.start == b.start;
    };
    const auto sameLine = [&sameMove](const std::vector<Move>& a,
                                      const std::vector<Move>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (!sameMove(a[i], b[i])) return false;
        return true;
    };
    const auto candidateBetter = [&](const RootPlan& existing) {
        return betterRootPlan(won, planScore, (size_t)node.depth, existing);
    };
    const auto it = destination.find(firstKey);
    const auto backup = backups.find(firstKey);
    // No line allocation/copy for the many children that cannot improve
    // either retained continuation. Reconstruct only published candidates.
    if (it != destination.end() && !candidateBetter(it->second) &&
        backup != backups.end() && !candidateBetter(backup->second)) return;
    std::vector<Move> line;
    if (!reconstructSearchLine(planner, node, line)) return;
    if (won && !retainPlannerWinningCertificate(planner,
            RootPlan{node.first, line, planScore, progress, true})) return;
    const auto retainCandidate = [&]() {
        return RootPlan{node.first, line, planScore, progress, won};
    };
    // For equally strong continuations keep the shorter clear. This reduces
    // physical actions without allowing a shallow but strategically weaker
    // line to beat a higher-scoring proven route. Retain one distinct runner-up
    // so certification can fall back without restarting an exhausted search.
    if (it == destination.end()) {
        destination.emplace(firstKey, retainCandidate());
    } else if (sameLine(line, it->second.line)) {
        if (candidateBetter(it->second)) it->second = retainCandidate();
    } else if (candidateBetter(it->second)) {
        backups[firstKey] = std::move(it->second);
        it->second = retainCandidate();
    } else {
        if (backup == backups.end()) {
            backups.emplace(firstKey, retainCandidate());
        } else if (candidateBetter(backup->second)) {
            backup->second = retainCandidate();
        }
    }
    if (provenProgress) planner.foundProgress = true;
}

struct StockSearchPrefix {
    SearchBoard board;
    uint8_t flips = 0;
    int tacticalCost = 0;
};
struct SearchEdge {
    Move move;
    int prefix = -1;
    // A stock macro hides its useful terminal action behind one or more Talon
    // clicks. Keep that tempo visible to best-first ordering: otherwise an Ace
    // in the stock looks like a sequence of low-value draw actions and can be
    // rediscovered only after another complete pass.
    int tempoBonus = 0;
};

// Reuse the existing monotone endgame prover INSIDE the existing planner.
// Once a searched bridge opens a complete Foundation/stock finish, retain
// the whole certificate rather than rediscovering it at every next root.
// A blocked monotone suffix is NOT a dead board: normal search continues.
static bool tryPlannerFoundationFinish(PersistentExpertPlanner& planner,
                                       const PersistentNode& node,
                                       const GameState& state, uint64_t deadline) {
    if (!planner.perfectInformation || node.path == SOL_NO_PARENT || node.hiddenCount > 6 ||
        node.foundationCount == 52 || nowMs() >= deadline) return false;
    const auto key = searchPositionKey(node.board, planner.deck, true);
    if (planner.blockedFoundationSuffixes.count(key)) return false;
    FoundationFinishProbe probe;
    probe.cursor = state;
    probe.deck = planner.deck;
    probe.board = node.board;
    probe.compactReady = true;
    probe.line.reserve(96);
    probe.visited.reserve(128);
    const auto result = advanceFoundationOnlyFinishPlan(probe, 224, deadline);
    if (result == FoundationFinishResult::Blocked) {
        if (planner.blockedFoundationSuffixes.size() < 512)
            planner.blockedFoundationSuffixes.insert(key);
        return false;
    }
    if (result != FoundationFinishResult::Won || probe.line.empty() ||
        node.depth + (int)probe.line.size() > planner.maxDepth ||
        planner.paths.size() + probe.line.size() > (size_t)planner.environment.arenaNodes * 4)
        return false;
    PersistentNode finished = node;
    int draws = 0;
    for (const Move& move : probe.line) {
        const uint32_t parent = finished.path;
        finished.path = (uint32_t)planner.paths.size();
        planner.paths.push_back({parent, move, 0});
        ++finished.depth;
        draws += move.kind == VirtualMoveKind::Talon;
    }
    finished.board = probe.board;
    finished.hiddenCount = 0;
    finished.foundationCount = 52;
    finished.autoCompleteProven = false;
    finished.structuralScore = virtualStructuralScore(probe.cursor);
    const int paid = std::max(0, probe.board.recycles - SOL_FREE_TALON_RECYCLES) -
                     std::max(0, node.board.recycles - SOL_FREE_TALON_RECYCLES);
    finished.accumulated -= draws * SOL_TALON_STEP_SEARCH_COST +
                            paid * SOL_CHARGED_RECYCLE_SEARCH_COST;
    finished.priority = finished.structuralScore + finished.accumulated - finished.depth * 18 + 2000000;
    recordRootPlan(planner, finished, probe.cursor);
    return planner.solved;
}

// K+-style additional edges, not a replacement set of rules. Ordinary stock
// edges are removed only after a complete known orbit. A macro reaches an exposed Waste card after
// exact Draw-N/recycle transitions and then plays it. Its path link retains
// EACH real stock action, physical depth and charged recycle cost.
static void generateStockSearchEdges(const GameState& state, const SearchDeck& deck,
                                      bool pullback, bool distantKing, bool compressKnownOrbit,
                                      uint64_t deadline,
                                      std::vector<Move>& ordinary,
                                      std::vector<SearchEdge>& edges,
                                      std::vector<StockSearchPrefix>& prefixes,
                                      GameState& cursor,
                                      StockOrbitCache* cache = nullptr) {
    generateVirtualMoves(state, ordinary, pullback, distantKing);
    edges.clear();
    prefixes.clear();
    for (const Move& move : ordinary) edges.push_back({move, -1, 0});
    const int stockCards = (int)state.talon.size() + (int)state.waste.size();
    if (stockCards <= 3) return;
    (void)cursor; // scratch argument retained for existing callers
    SearchBoard root;
    if (!packSearchBoard(state, deck, root)) return;
    SearchBoard orbit = root;
    const Move draw{VirtualMoveKind::Talon, -1, -1, -1};
    // Values and Tableau/Foundation are immutable during a stock-only orbit.
    // Compare exact token order, Waste visibility and split, not a hash. A
    // known-but-not-yet-visible Waste must not hide a newly playable exit. No
    // full-board key construction or unordered_set allocation for every flip.
    std::array<StockOrbitKey, 108> visited{};
    size_t visitedCount = 1;
    const StockOrbitKey initial = stockOrbitKey(root);
    const StockOrbitKey cacheKey = stockOrbitKey(root, true);
    visited[0] = initial;
    const int maxFlips = 2 * ((stockCards + std::clamp(state.drawCount, 1, 3) - 1) /
                              std::clamp(state.drawCount, 1, 3)) + 2;
    bool completeOrbit = false;
    const std::vector<StockOrbitStep>* cached = nullptr;
    if (cache) {
        const auto found = cache->find(cacheKey);
        if (found != cache->end()) cached = &found->second;
    }
    std::array<StockOrbitStep, 108> route{};
    size_t routeSize = 0;
    unsigned stockBegin = 0;
    for (int pile = 0; pile < 11; ++pile) stockBegin += root.sizes[pile];
    for (int flips = 1; flips <= maxFlips && nowMs() < deadline; ++flips) {
        StockOrbitStep step;
        if (cached) {
            if ((size_t)(flips - 1) == cached->size()) { completeOrbit = true; break; }
            step = (*cached)[flips - 1];
            orbit = root;
            orbit.sizes[11] = step.position.stock;
            orbit.sizes[12] = step.position.waste;
            orbit.recycles += step.extraRecycles;
            orbit.stagnantRecycles += step.extraRecycles;
            std::copy_n(step.tokens.begin(), stockCards, orbit.cards.begin() + stockBegin);
        } else {
            if (!applySearchMove(orbit, deck, draw)) break;
            step.position = stockOrbitKey(orbit);
            std::copy_n(orbit.cards.begin() + stockBegin, stockCards, step.tokens.begin());
            step.flips = (uint8_t)flips;
            step.extraRecycles = (uint8_t)(orbit.recycles - root.recycles);
            bool repeated = false;
            for (size_t i = 0; i < visitedCount; ++i)
                if (visited[i] == step.position) { repeated = true; break; }
            if (repeated) { completeOrbit = true; break; }
            if (visitedCount == visited.size()) break;
            visited[visitedCount++] = step.position;
            route[routeSize++] = step;
        }
        if (!orbit.sizes[12]) continue;
        const uint8_t exposedToken = orbit.cards[stockBegin + stockCards - 1];
        if (!(exposedToken & 0x40)) continue;
        const int index = (int)prefixes.size();
        StockSearchPrefix prefix;
        prefix.board = orbit;
        prefix.flips = (uint8_t)flips;
        const int paid = std::max(0, orbit.recycles - SOL_FREE_TALON_RECYCLES) -
                         std::max(0, state.recycles - SOL_FREE_TALON_RECYCLES);
        prefix.tacticalCost = -SOL_TALON_STEP_SEARCH_COST * flips -
                              paid * SOL_CHARGED_RECYCLE_SEARCH_COST;
        Card exposed;
        exposed.id = deck.cards[exposedToken & 0x3f].id;
        exposed.value = exposed.knownValue = deck.cards[exposedToken & 0x3f].knownValue;
        const size_t previousEdges = edges.size();
        for (int dst = 0; dst < 4; ++dst) {
            const Card* top = state.foundations[dst].empty() ? nullptr : &state.foundations[dst].back();
            if (canDropOnFoundation(exposed, top)) {
                const int rank = cardRank(exposed);
                const bool safe = rank <= 1 ||
                    foundationAdvanceIsSafe(exposed, state.foundations);
                const int tempo = rank <= 1 ? 90000 - flips * 1200 :
                                  (safe ? 18000 - flips * 450 : 0);
                edges.push_back({{VirtualMoveKind::Foundation, -1, dst, -1},
                                 index, std::max(0, tempo)});
            }
        }
        for (int dst = 0; dst < 7; ++dst) {
            const Card* top = state.tableau[dst].empty() ? nullptr : &state.tableau[dst].back();
            if (canDropOnTableau(exposed, top))
                edges.push_back({{VirtualMoveKind::Tableau, -1, dst, -1},
                                 index, std::max(0, 5000 - flips * 300)});
        }
        if (edges.size() != previousEdges) prefixes.push_back(prefix);
    }
    if (!cached && cache && completeOrbit && cache->size() < 64)
        cache->emplace(cacheKey, std::vector<StockOrbitStep>(route.begin(), route.begin() + routeSize));
    if (compressKnownOrbit && completeOrbit) {
        // Tableau/Foundation actions commute with stock-only flips: execute
        // them first, then examine the new stock orbit with their new targets.
        // A complete known orbit's useful Waste exits are all represented by
        // macros. Pure stock laps have positive cost and create no progress.
        // If knowledge or the time budget is incomplete, retain ordinary
        // Talon edges instead; a partial scan must NEVER remove a real route.
        edges.erase(std::remove_if(edges.begin(), edges.end(), [](const SearchEdge& edge) {
            return edge.prefix < 0 && edge.move.kind == VirtualMoveKind::Talon;
        }), edges.end());
    }
}

static void advancePlannerSlice(PersistentExpertPlanner& planner,
                                 const GameState& rootState) {
    ++planner.frames;
    auto& env = planner.environment;
    const int frameNodeBudget = env.recovery
        ? (planner.endgame ? env.sliceNodes * 2 : env.sliceNodes + env.sliceNodes / 2)
        : (planner.endgame ? env.sliceNodes + env.sliceNodes / 2 : env.sliceNodes);
    const int frameSliceMs = env.sliceMs + (planner.endgame ? 2 : 0) +
                             (env.recovery ? 2 : 0);
    const uint64_t deadline = nowMs() + (uint64_t)frameSliceMs;
    const bool allowOpaqueRootPullback = env.recovery && !planner.perfectInformation &&
        env.escalation >= SOL_MAX_RECOVERY_ESCALATION && env.movesWithoutProgress >= 8;
    static thread_local GameState state, prefixState, next, macroCursor;
    static thread_local std::vector<Move> moves;
    static thread_local std::vector<SearchEdge> edges;
    static thread_local std::vector<StockSearchPrefix> prefixes;
    static thread_local bool reserved = false;
    if (!reserved) {
        reserveSearchScratch(state);
        reserveSearchScratch(prefixState);
        reserveSearchScratch(next);
        reserveSearchScratch(macroCursor);
        moves.reserve(64);
        edges.reserve(192);
        prefixes.reserve(40);
        reserved = true;
    }
    const auto failed = [&](const Move& first) {
        const std::string key = moveKey(first);
        return env.failedFirstEdges.count(key) || planner.uncertifiableFirstEdges.count(key);
    };
    const size_t pathLimit = (size_t)env.arenaNodes * 4;
    int expanded = 0;
    while (!planner.frontier.empty() && !planner.solved && !planner.capacityReached &&
           expanded < frameNodeBudget && nowMs() < deadline) {
        // Two actual expansions per first-action alternative prevent one
        // attractive shallow branch from monopolising all initial work. After
        // this bounded warm-up, the unchanged best-first priority takes over.
        PersistentNode node = env.balanceRootAlternatives && planner.rootWarmup
            ? planner.frontier.takeBalancedRoot(planner.rootCoverage, planner.rootWarmup)
            : planner.frontier.takeTop();
        const bool atRoot = node.path == SOL_NO_PARENT;
        if (!atRoot && failed(node.first)) continue;
        if (!unpackSearchBoard(node.board, planner.deck, state)) {
            planner.capacityReached = true; // invalid encoding is NOT a dead proof
            break;
        }
        ++expanded;
        ++planner.expanded;
        if (node.foundationCount == 52) {
            node.priority += 2000000;
            recordRootPlan(planner, node, state);
            if (planner.solved) break;
            continue; // invalid certificate is not a solved/dead-board proof
        }
        if (!planner.rootAutoCompleteLayout && node.autoCompleteProven) {
            recordRootPlan(planner, node, state);
            continue;
        }
        // Bounded additional proof work; do not monopolise a Unity frame.
        if (env.proveFoundationSuffix && (planner.expanded & 15) == 0 &&
            tryPlannerFoundationFinish(planner, node, state, deadline)) break;
        if (node.depth >= planner.maxDepth) continue;
        const bool widened = planner.perfectInformation &&
            (env.recovery || planner.widenedProofGraph);
        if (env.stockMacros) {
            generateStockSearchEdges(state, planner.deck,
                widened || (allowOpaqueRootPullback && atRoot), widened,
                planner.perfectInformation, deadline,
                moves, edges, prefixes, macroCursor, &planner.stockOrbitCache);
        } else {
            generateVirtualMoves(state, moves, widened || (allowOpaqueRootPullback && atRoot), widened);
            edges.clear(); prefixes.clear();
            for (const Move& move : moves) edges.push_back({move, -1, 0});
        }
        const std::string inherited = atRoot ? std::string() : moveKey(node.first);
        int decodedPrefix = -2;
        for (const SearchEdge& edge : edges) {
            // A generation can produce many siblings; honor the wall-clock
            // budget inside the loop too, not only between popped nodes.
            if (nowMs() >= deadline) {
                // Keep the node, since its remaining ordinary legal edges have
                // not been explored. Transposition will reject finished work.
                planner.frontier.push(node);
                break;
            }
            const int flips = edge.prefix < 0 ? 0 : prefixes[edge.prefix].flips;
            const Move first = atRoot
                ? (flips ? Move{VirtualMoveKind::Talon, -1, -1, -1} : edge.move)
                : node.first;
            if (failed(first)) continue;
            const int nextDepth = node.depth + flips + 1;
            if (nextDepth > planner.maxDepth) continue;
            if (edge.prefix != decodedPrefix) {
                if (edge.prefix < 0) prefixState = state;
                else if (!unpackSearchBoard(prefixes[edge.prefix].board, planner.deck,
                                            prefixState)) continue;
                decodedPrefix = edge.prefix;
            }
            SearchBoard nextBoard = edge.prefix < 0 ? node.board : prefixes[edge.prefix].board;
            if (!applySearchMove(nextBoard, planner.deck, edge.move)) continue;
            const SearchPositionKey compactKey = searchPositionKey(nextBoard, planner.deck);
            const std::string firstEdge = atRoot ? moveKey(first) : inherited;
            const SearchVisitKey visitKey{compactKey, searchMoveId(first), planner.widenedProofGraph};
            const int spent = std::max(0, nextBoard.recycles - rootState.recycles);
            const int paid = std::max(0, nextBoard.recycles - SOL_FREE_TALON_RECYCLES) -
                             std::max(0, rootState.recycles - SOL_FREE_TALON_RECYCLES);
            const int cost = nextDepth * 18 + spent * 120 +
                             paid * SOL_CHARGED_RECYCLE_SEARCH_COST;
            const auto visited = planner.bestPathCost.find(visitKey);
            if (visited != planner.bestPathCost.end() && visited->second <= cost) continue;
            // Decode only new/non-dominated states, not duplicate children.
            if (!unpackSearchBoard(nextBoard, planner.deck, next)) continue;
            std::string nextKey;
            if (!env.deadBoards.empty() || !env.strategicFailures.empty())
                nextKey = solverPositionKey(next);
            if (!env.deadBoards.empty() && env.deadBoards.count(nextKey)) continue;
            if (planner.paths.size() >= pathLimit ||
                planner.frontier.size() >= (size_t)env.arenaNodes) {
                planner.capacityReached = true;
                break; // resource limit is never cached as an impossible state
            }
            PersistentNode child;
            child.hiddenCount = node.hiddenCount - virtualRevealGain(prefixState, edge.move);
            child.foundationCount = node.foundationCount +
                (edge.move.kind == VirtualMoveKind::Foundation ? 1 :
                 (edge.move.kind == VirtualMoveKind::FoundationToTableau ? -1 : 0));
            // Conservative proof only in small, fully known states. This is
            // independent of heuristic score and admits grouped suffix moves.
            // Run the sound delete-relaxation at every shallow root fork and
            // throughout the proof-first endgame.  V14 deferred this until
            // seven hidden cards remained, which could discover that an
            // earlier route was impossible only after the decisive card had
            // already been committed physically.  The shallow check rejects
            // that bad first route while the wider threshold protects the
            // complete endgame without charging every early-game node.
            if (planner.perfectInformation && child.foundationCount < 52 &&
                (nextDepth <= 2 || child.hiddenCount <= SOL_VICTORY_ENDGAME_HIDDEN)) {
                const auto cachedProof = planner.dependencyProofCache.find(compactKey);
                const bool dead = cachedProof != planner.dependencyProofCache.end()
                    ? cachedProof->second : provenDependencyDeadEnd(next);
                if (cachedProof == planner.dependencyProofCache.end() &&
                    planner.dependencyProofCache.size() < SOL_STRUCTURAL_CACHE_LIMIT)
                    planner.dependencyProofCache.emplace(compactKey, dead);
                if (dead) {
                    if (nextKey.empty()) nextKey = solverPositionKey(next);
                    env.deadBoards.insert(nextKey);
                    env.newDeadBoards.push_back(nextKey);
                    continue;
                }
            }
            child.board = nextBoard;
            child.depth = nextDepth;
            child.first = first;
            child.autoCompleteProven = child.hiddenCount == 0 &&
                child.foundationCount < 52 && provenAutoCompleteLayout(next);
            child.accumulated = node.accumulated +
                (edge.prefix < 0 ? 0 : prefixes[edge.prefix].tacticalCost) +
                edge.tempoBonus +
                candidateTacticalBonus(prefixState, next, edge.move, node.hiddenCount,
                                       child.autoCompleteProven ? 1 : 0);
            const auto cached = planner.structuralScoreCache.find(compactKey);
            if (cached != planner.structuralScoreCache.end()) child.structuralScore = cached->second;
            else {
                child.structuralScore = virtualStructuralScore(next, child.autoCompleteProven ? 1 : 0);
                if (planner.structuralScoreCache.size() < SOL_STRUCTURAL_CACHE_LIMIT)
                    planner.structuralScoreCache.emplace(compactKey, child.structuralScore);
            }
            child.priority = child.structuralScore + child.accumulated - nextDepth * 18;
            if (env.recentBoards.count(recentPositionKey(next))) child.priority -= 4000000;
            const auto failure = env.strategicFailures.find(nextKey);
            if (failure != env.strategicFailures.end())
                child.priority -= std::min(1400000, failure->second *
                                           (env.recovery ? 280000 : 65000));
            if (child.foundationCount == 52) child.priority += 2000000;
            child.path = (uint32_t)planner.paths.size();
            planner.paths.push_back({node.path, edge.move, (uint8_t)flips});
            planner.bestPathCost[visitKey] = cost;
            if (atRoot) {
                planner.rootCoverage.emplace(searchMoveId(first), 0);
                planner.rootWarmup = true;
            }
            int& horizon = planner.rootHorizon[firstEdge];
            horizon = std::max(horizon, nextDepth);
            recordRootPlan(planner, child, next);
            if (planner.solved) break;
            if (child.foundationCount == 52) continue;
            planner.frontier.push(std::move(child));
        }
        bool terminal = moves.empty() && node.foundationCount < 52 && !hasOpaqueCards(state);
        if (terminal && !widened) {
            generateVirtualMoves(state, moves, true, true);
            terminal = moves.empty();
        }
        if (terminal) {
            const std::string key = solverPositionKey(state);
            if (env.deadBoards.insert(key).second) env.newDeadBoards.push_back(key);
            if (!atRoot) {
                int& leaves = planner.rootTerminalLeaves[moveKey(node.first)];
                leaves = std::min(64, leaves + 1);
            }
        }
    }
}

static void advancePersistentSearch(const GameState& rootState) {
    g_accessStage.store(41, std::memory_order_release);
    auto& env = g_persistentPlanner.environment;
    // Current root legality/flight guards are evaluated only on Unity. Worker
    // routes remain virtual and undergo these same guards after handoff.
    env.recovery = g_recoveryMode;
    env.recoveryPreparations = g_progressTracker.recoveryPreparations;
    env.movesWithoutProgress = g_progressTracker.movesWithoutProgress;
    env.baselineHidden = g_progressTracker.initialized ? g_progressTracker.baseline.hidden : hiddenCardCount(rootState);
    env.baselineFoundation = g_progressTracker.initialized ? g_progressTracker.baseline.foundation : virtualFoundationCount(rootState);
    env.baselineDebt = g_progressTracker.initialized ? g_progressTracker.baseline.foundationDebt : foundationRiskDebt(rootState.foundations);
    env.sliceNodes = g_searchSliceNodes.load(std::memory_order_relaxed);
    env.sliceMs = g_searchSliceMs.load(std::memory_order_relaxed);
    static thread_local std::vector<Move> rootMoves;
    generateVirtualMoves(rootState, rootMoves, true, true);
    for (const Move& move : rootMoves) {
        const auto key = moveKey(move);
        if (rootMoveFailed(g_persistentPlanner.rootKey, move) ||
            !physicalCandidateAllowed(rootState, move, nowMs())) env.failedFirstEdges.insert(key);
        else env.failedFirstEdges.erase(key); // temporary guard is not a permanent branch failure
    }
    advancePlannerSlice(g_persistentPlanner, rootState);
    for (const auto& key : env.newDeadBoards) g_deadStateCache.insert(key);
    env.newDeadBoards.clear();
    g_accessStage.store(42, std::memory_order_release);
}

struct AnticipatoryJob {
    GameState state;
    SearchEnvironment environment;
    std::string key;
    uint64_t generation = 0;
};
struct AnticipatoryResult {
    PersistentExpertPlanner planner;
    uint64_t generation = 0;
    bool ready = false;
};

// One worker runs THE SAME advancePlannerSlice, not a competing solver.
// It owns every container it mutates. No IL2CPP pointer, JNI call or physical
// guard crosses this boundary; cancellation is checked between short slices.
class AnticipatoryWorker {
public:
    std::mutex mutex;
    std::condition_variable wake;
    std::atomic<uint64_t> generation{1};
    // Published only after the per-job `current` snapshot is constructed and
    // cleared only after that snapshot and its planner allocations die. The
    // Unity thread uses this as the safe point for a between-deals heap purge.
    std::atomic<bool> busy{false};
    // When the physical action settles before the 190 ms opportunity window,
    // Unity requests the frontier and the worker publishes after its current
    // bounded slice instead of making the owner start a duplicate search.
    std::atomic<bool> handoffRequested{false};
    uint64_t activeGeneration = 0; // guarded by mutex
    std::string activeKey;         // guarded by mutex
    bool stopping = false, hasJob = false;
    AnticipatoryJob job;
    AnticipatoryResult result;
    std::thread thread;

    AnticipatoryWorker() : thread([this] { run(); }) {}
    ~AnticipatoryWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            generation.fetch_add(1, std::memory_order_relaxed);
        }
        wake.notify_one();
        thread.join();
    }
private:
    struct BusyScope {
        std::atomic<bool>& flag;
        bool armed = false;
        ~BusyScope() {
            if (armed) flag.store(false, std::memory_order_release);
        }
    };

    void run() {
        for (;;) {
            // Declared before `current`, so reverse destruction destroys all
            // job-owned frontier memory before publishing busy=false.
            BusyScope busyScope{busy};
            AnticipatoryJob current;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || hasJob; });
                if (stopping) return;
                current = std::move(job);
                hasJob = false;
                // Publish ownership while holding the same mutex used by
                // cancellation; there is then no gap where `current` owns an
                // old frontier but the Unity thread observes busy=false.
                busy.store(true, std::memory_order_release);
                busyScope.armed = true;
                activeGeneration = current.generation;
                activeKey = current.key;
                handoffRequested.store(false, std::memory_order_release);
            }
            try {
                AnticipatoryResult completed;
                completed.generation = current.generation;
                if (!initializePlannerRoot(completed.planner, current.state, current.key,
                                            std::move(current.environment))) continue;
                std::vector<Move> finish;
                if (hiddenCardCount(current.state) <= SOL_VICTORY_ENDGAME_HIDDEN &&
                    findFoundationOnlyFinishPlan(current.state, finish) &&
                    verifiedWinningLine(current.state, finish)) {
                    RootPlan win{finish.front(), std::move(finish), 4000000, 2000000, true};
                    if (retainPlannerWinningCertificate(completed.planner, win))
                        completed.planner.roots.emplace(moveKey(win.first), std::move(win));
                } else {
                    // Spend the real animation/server-ACK window on proof work.
                    // V15 stopped after roughly 72 ms and left the remaining
                    // flight idle; RAPTOR continues for a bounded 190 ms, then
                    // transfers even an unfinished frontier back intact.
                    const uint64_t prefetchDeadline =
                        nowMs() + SOL_RAPTOR_PREFETCH_MS;
                    const int prefetchSlices = std::clamp(
                        (int)(SOL_RAPTOR_PREFETCH_MS /
                              std::max(1, completed.planner.environment.sliceMs)),
                        16, SOL_RAPTOR_PREFETCH_MAX_SLICES);
                    for (int slice = 0; slice < prefetchSlices && !completed.planner.solved &&
                         !completed.planner.frontier.empty() &&
                         !completed.planner.capacityReached &&
                         nowMs() < prefetchDeadline &&
                         generation.load(std::memory_order_relaxed) == current.generation; ++slice) {
                        advancePlannerSlice(completed.planner, current.state);
                        if (handoffRequested.load(std::memory_order_acquire)) break;
                    }
                }
                completed.ready = true;
                AnticipatoryResult retired;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (generation.load(std::memory_order_relaxed) != current.generation) continue;
                    retired = std::move(result);
                    result = std::move(completed);
                    activeGeneration = 0;
                    activeKey.clear();
                    handoffRequested.store(false, std::memory_order_release);
                }
                // Releasing obsolete frontiers does not hold the handoff lock.
            } catch (...) {
                // Allocation failure must never escape the worker and abort
                // the process. The unchanged owner-thread path remains usable.
            }
        }
    }
};

static AnticipatoryWorker& anticipatoryWorker() {
    static AnticipatoryWorker worker;
    return worker;
}
// Do not start a worker simply to reset the disabled engine during launch.
static AnticipatoryWorker* g_startedAnticipatoryWorker = nullptr;
// Owner-thread identity of the queued, running, or ready route-tail job. This
// prevents each intermediate click in the same committed prefix from cancelling
// and restarting an identical destination search.
static std::string g_anticipatoryTargetKey;

static void cancelAnticipatorySearch() {
    if (!g_startedAnticipatoryWorker) return;
    auto& worker = *g_startedAnticipatoryWorker;
    worker.generation.fetch_add(1, std::memory_order_relaxed);
    AnticipatoryJob retiredJob;
    AnticipatoryResult retiredResult;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        worker.hasJob = false;
        worker.handoffRequested.store(false, std::memory_order_release);
        retiredJob = std::move(worker.job);
        retiredResult = std::move(worker.result);
        worker.result.ready = false;
    }
    g_anticipatoryTargetKey.clear();
}

// Return false only for the few frames needed by a cancelled worker slice to
// unwind. This prevents a new deal from immediately allocating another 320k
// frontier on top of pages still owned by the previous deal.
static bool finishRoundMemoryReleaseIfReady(uint64_t now) {
    if (!g_roundMemoryReleasePending.load(std::memory_order_acquire)) return true;
    if (g_startedAnticipatoryWorker &&
        g_startedAnticipatoryWorker->busy.load(std::memory_order_acquire)) {
        return false;
    }

    using MallocTrimFn = int (*)(size_t);
    static MallocTrimFn mallocTrim = reinterpret_cast<MallocTrimFn>(
        dlsym(RTLD_DEFAULT, "malloc_trim"));
    const int purged = mallocTrim ? mallocTrim(0) : -1;
    g_roundMemoryReleasePending.store(false, std::memory_order_release);
    g_lastRoundMemoryReleaseAt = now;
    SOL_LOG("[ROUND_MEMORY] released requestedAge=%llu allocatorPurge=%d",
            (unsigned long long)(now - g_roundMemoryReleaseRequestedAt), purged);
    return true;
}

static void scheduleAnticipatorySearch(const GameState& expected) {
    // When a certified prefix has more actions, the next click is already
    // known. Replay that remainder virtually and solve the prefix destination
    // while the phone executes it. This removes the old think-pause after each
    // reveal/Foundation milestone without ever dispatching an unacknowledged
    // action. If the physical route diverges, its exact key will not be accepted.
    GameState target = expected;
    for (const Move& step : g_committedPlan)
        if (!applyVirtualMove(target, step)) return;
    if (!completeKnownDeck(target) || virtualFoundationCount(target) == 52) return;
    auto& worker = anticipatoryWorker();
    g_startedAnticipatoryWorker = &worker;
    const std::string targetKey = virtualPositionKey(target);
    if (g_anticipatoryTargetKey == targetKey) return;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        const uint64_t live = worker.generation.load(std::memory_order_relaxed);
        if ((worker.hasJob && worker.job.generation == live &&
             worker.job.key == targetKey) ||
            (worker.result.ready && worker.result.generation == live &&
             worker.result.planner.rootKey == targetKey)) return;
    }
    AnticipatoryJob next;
    next.state = std::move(target);
    next.key = targetKey;
    next.environment = captureSearchEnvironment(next.state, next.key);
    const auto progress = durableProgressFrom(next.state, next.environment.baselineHidden,
        next.environment.baselineFoundation, next.environment.baselineDebt);
    if (progress.any()) {
        next.environment.baselineHidden = hiddenCardCount(next.state);
        next.environment.baselineFoundation = virtualFoundationCount(next.state);
        next.environment.baselineDebt = foundationRiskDebt(next.state.foundations);
        next.environment.recovery = false;
        next.environment.recoveryPreparations = 0;
        next.environment.movesWithoutProgress = 0;
    }
    next.generation = worker.generation.fetch_add(1, std::memory_order_relaxed) + 1;
    AnticipatoryJob retired;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        retired = std::move(worker.job);
        worker.job = std::move(next);
        worker.hasJob = true;
        worker.result.ready = false;
        worker.handoffRequested.store(false, std::memory_order_release);
    }
    g_anticipatoryTargetKey = targetKey;
    worker.wake.notify_one();
}

static bool takeAnticipatorySearch(const GameState& observed, const std::string& key) {
    if (!g_startedAnticipatoryWorker || g_waitingForMoveResult) return false;
    auto& worker = *g_startedAnticipatoryWorker;
    AnticipatoryResult ready;
    bool handoffPending = false;
    {
        std::lock_guard<std::mutex> lock(worker.mutex);
        const auto& candidate = worker.result;
        const uint64_t live = worker.generation.load(std::memory_order_relaxed);
        const bool candidateMatches = candidate.ready &&
            candidate.generation == live && candidate.planner.rootKey == key &&
            candidate.planner.rootRecycles == observed.recycles;
        if (!candidateMatches) {
            if (worker.busy.load(std::memory_order_acquire) &&
                worker.activeGeneration == live && worker.activeKey == key) {
                worker.handoffRequested.store(true, std::memory_order_release);
                handoffPending = true;
            } else if (worker.hasJob && worker.job.generation == live &&
                       worker.job.key == key) {
                handoffPending = true;
            } else {
                return false;
            }
        }
        if (!handoffPending) {
        const auto& env = candidate.planner.environment;
        if (g_progressTracker.initialized &&
            (env.baselineHidden != g_progressTracker.baseline.hidden ||
             env.baselineFoundation != g_progressTracker.baseline.foundation ||
             env.baselineDebt != g_progressTracker.baseline.foundationDebt ||
             env.recovery != g_recoveryMode)) return false;
        ready = std::move(worker.result);
        worker.result.ready = false;
        }
    }
    if (handoffPending) {
        g_dispatchDeferredThisTick = true;
        return false;
    }
    g_persistentPlanner = std::move(ready.planner);
    g_anticipatoryTargetKey.clear();
    for (const auto& dead : g_persistentPlanner.environment.newDeadBoards) g_deadStateCache.insert(dead);
    g_persistentPlanner.environment.newDeadBoards.clear();
    SOL_LOG("[SEARCH] accepted-animation-prefetch frames=%d expanded=%d exactAck=1",
            g_persistentPlanner.frames, g_persistentPlanner.expanded);
    return true;
}

static int requiredRootHorizon(const PersistentExpertPlanner& planner) {
    if (!planner.perfectInformation) return 5;
    if (planner.endgame) return 48;
    if (planner.rootHidden <= 18) return 32;
    return 20;
}

static bool widenStalledProofSearch(const GameState& state,
                                   const std::string& rootKey, uint64_t now) {
    auto& planner = g_persistentPlanner;
    if (g_recoveryMode || planner.widenedProofGraph || !planner.perfectInformation ||
        planner.solved || !planner.roots.empty() || planner.frontier.empty() ||
        g_waitingForMoveResult || !g_committedPlan.empty() ||
        !g_progressTracker.initialized || g_progressTracker.stallPending ||
        planner.rootKey != rootKey || g_progressTracker.exactKey != rootKey)
        return false;
    const uint64_t changedAt = g_progressTracker.stateChangedAt;
    const uint64_t idle = now >= changedAt ? now - changedAt : 0;
    const bool enoughWork = planner.frames >= 8 &&
        planner.expanded >= (planner.endgame ? 1500 : 1000);
    const bool slowDeviceFallback = planner.frames >= 16 && idle >= 1200;
    if ((!enoughWork || idle < (planner.endgame ? 500ULL : 350ULL)) &&
        !slowDeviceFallback) return false;

    // Widen the existing search before the 6.5/9-second strategic-failure
    // timer. Preserve its frontier, checkpoints and history. This only adds
    // virtual options: physical execution still requires proven progress;
    // cosmetic recovery preparations remain disabled until actual recovery.
    planner.widenedProofGraph = true;
    planner.maxDepth = std::max(planner.maxDepth, planner.endgame ? 288 : 208);
    PersistentNode root;
    if (!packSearchBoard(state, planner.deck, root.board)) return false;
    root.hiddenCount = planner.rootHidden;
    root.foundationCount = planner.rootFoundation;
    root.autoCompleteProven = planner.rootAutoCompleteLayout;
    root.structuralScore = planner.rootStructuralScore;
    // Expand new root edges on the next slice; child scores remain unchanged.
    root.priority = std::numeric_limits<int>::max();
    planner.frontier.push(std::move(root));
    SOL_LOG("[RECOVERY] early-proof-widen idle=%llu expanded=%d frontier=%d",
            (unsigned long long)idle, planner.expanded, (int)planner.frontier.size());
    return true;
}

static int rootHorizonFor(const Move& move) {
    const auto it = g_persistentPlanner.rootHorizon.find(moveKey(move));
    return it == g_persistentPlanner.rootHorizon.end() ? 0 : it->second;
}

static int rootTerminalLeavesFor(const Move& move) {
    const auto it = g_persistentPlanner.rootTerminalLeaves.find(moveKey(move));
    return it == g_persistentPlanner.rootTerminalLeaves.end() ? 0 : it->second;
}

static int bestExecutableRootHorizon(const std::string& rootKey) {
    int best = 0;
    const auto consider = [&](const auto& plans) {
        for (const auto& entry : plans) {
            if (rootMoveFailed(rootKey, entry.second.first)) continue;
            best = std::max(best, rootHorizonFor(entry.second.first));
        }
    };
    consider(g_persistentPlanner.roots);
    if (g_recoveryMode) consider(g_persistentPlanner.recoveryRoots);
    return best;
}

static void discardPlannerRootEdge(const std::string& rootKey,
                                   const std::string& failedMoveKey) {
    if (g_persistentPlanner.rootKey != rootKey) return;
    if (g_persistentPlanner.hasWinningCertificate &&
        moveKey(g_persistentPlanner.winningCertificate.first) == failedMoveKey) {
        g_persistentPlanner.hasWinningCertificate = false;
        g_persistentPlanner.solved = false;
        g_persistentPlanner.winningCertificate = RootPlan{};
    }
    g_persistentPlanner.roots.erase(failedMoveKey);
    g_persistentPlanner.rootBackups.erase(failedMoveKey);
    g_persistentPlanner.recoveryRoots.erase(failedMoveKey);
    g_persistentPlanner.recoveryRootBackups.erase(failedMoveKey);
    // Frontier nodes are intentionally retained. advancePersistentSearch()
    // lazily drops only nodes whose first edge is now failed, preserving every
    // still-valid sibling and all accumulated transposition work.
}

static bool promotePlannerRootBackup(const std::string& rootKey,
                                     const std::string& firstMoveKey,
                                     bool recoveryPlan) {
    if (g_persistentPlanner.rootKey != rootKey) return false;
    auto& destination = recoveryPlan ? g_persistentPlanner.recoveryRoots :
                                      g_persistentPlanner.roots;
    auto& backups = recoveryPlan ? g_persistentPlanner.recoveryRootBackups :
                                  g_persistentPlanner.rootBackups;
    const auto backup = backups.find(firstMoveKey);
    if (backup == backups.end()) {
        destination.erase(firstMoveKey);
        return false;
    }
    destination[firstMoveKey] = std::move(backup->second);
    backups.erase(backup);
    return true;
}

static void registerPhysicalRootFailure(const std::string& rootKey,
                                        const std::string& failedMoveKey) {
    g_failedRootMoves[rootKey].insert(failedMoveKey);
    discardPlannerRootEdge(rootKey, failedMoveKey);
}

static bool deferPhysicalDispatch(const GameState& state) {
    if (g_lastDispatchOutcome != DispatchOutcome::Deferred) return false;
    g_dispatchRetryPhysicalKey = physicalPositionKey(state);
    return true;
}

static std::vector<std::string> legalAlternativeKeys(const GameState& state,
                                                     const Move& selected) {
    std::vector<Move> legal;
    const bool perfectRecovery = g_recoveryMode &&
        knownCardCount(state) == 52 && !hasOpaqueCards(state);
    generateVirtualMoves(state, legal, perfectRecovery, perfectRecovery);
    std::vector<std::string> alternatives;
    const std::string selectedKey = moveKey(selected);
    const std::string rootKey = virtualPositionKey(state);
    alternatives.reserve(legal.size());
    for (const Move& move : legal) {
        const std::string key = moveKey(move);
        if (key != selectedKey && !rootMoveFailed(rootKey, move))
            alternatives.push_back(key);
    }
    std::sort(alternatives.begin(), alternatives.end());
    alternatives.erase(std::unique(alternatives.begin(), alternatives.end()),
                       alternatives.end());
    return alternatives;
}

static uint64_t saveDecisionCheckpoint(const GameState& root,
                                       const Move& selected,
                                       const std::vector<Move>& selectedLine,
                                       std::vector<std::string> alternatives,
                                       int score) {
    // A single forced move has no branch to backtrack.  A multi-step committed
    // route is still important because its later virtual states can be used as
    // strategic-failure evidence even if the root had no immediate sibling.
    if (alternatives.empty() && selectedLine.size() <= 1) return 0;
    const std::string rootExact = virtualPositionKey(root);
    const std::string selectedKey = moveKey(selected);
    if (!g_decisionCheckpoints.empty()) {
        const DecisionCheckpoint& last = g_decisionCheckpoints.back();
        if (!last.failed && last.rootExactKey == rootExact &&
            last.selectedMoveKey == selectedKey) return 0;
    }

    DecisionCheckpoint checkpoint;
    checkpoint.id = g_nextCheckpointId++;
    checkpoint.progressEpoch = g_progressTracker.progressEpoch;
    checkpoint.createdAt = nowMs();
    checkpoint.moveSerial = g_progressTracker.moveSerial;
    checkpoint.virtualRoot = root;
    checkpoint.metrics = g_progressTracker.initialized &&
                         g_progressTracker.exactKey == rootExact ?
                         g_progressTracker.current :
                         collectProgressMetrics(root, (int)selectedLine.size());
    checkpoint.rootExactKey = rootExact;
    checkpoint.rootSolverKey = solverPositionKey(root);
    checkpoint.selected = selected;
    checkpoint.selectedMoveKey = selectedKey;
    checkpoint.selectedLine = selectedLine;
    checkpoint.alternatives = std::move(alternatives);
    checkpoint.score = score;

    GameState cursor = root;
    const int pathLimit = std::min(24, (int)selectedLine.size());
    for (int i = 0; i < pathLimit; ++i) {
        if (!applyVirtualMove(cursor, selectedLine[i])) break;
        const std::string stateKey = solverPositionKey(cursor);
        checkpoint.pathStateKeys.push_back(stateKey);
        if (i == 0) checkpoint.selectedResultKey = stateKey;
    }
    if (checkpoint.selectedResultKey.empty()) {
        GameState first = root;
        if (applyVirtualMove(first, selected))
            checkpoint.selectedResultKey = solverPositionKey(first);
    }

    const uint64_t checkpointId = checkpoint.id;
    g_decisionCheckpoints.push_back(std::move(checkpoint));
    pruneOldCheckpoints();
    return checkpointId;
}

static void discardUnverifiedCheckpoint(uint64_t checkpointId) {
    if (checkpointId == 0 || g_decisionCheckpoints.empty()) return;
    if (g_decisionCheckpoints.back().id == checkpointId)
        g_decisionCheckpoints.pop_back();
}

static DecisionCheckpoint* latestRecoveryCheckpoint() {
    // Prefer the current no-progress epoch.  If every move since the last
    // irreversible gain was forced, allow the immediately preceding epoch's
    // latest decision: a superficially useful foundation/reveal move can still
    // be the strategic choice that closed the winning continuation.
    for (int pass = 0; pass < 2; ++pass) {
        const int wantedEpoch = g_progressTracker.progressEpoch - pass;
        if (wantedEpoch < 0) continue;
        for (auto it = g_decisionCheckpoints.rbegin();
             it != g_decisionCheckpoints.rend(); ++it) {
            if (it->failed || it->progressEpoch != wantedEpoch) continue;
            if (g_progressTracker.moveSerial - it->moveSerial > 28) continue;
            return &*it;
        }
    }
    return nullptr;
}

static int remainingCheckpointAlternatives(const DecisionCheckpoint& checkpoint) {
    int remaining = 0;
    const auto failed = g_failedStrategicBranches.find(checkpoint.rootExactKey);
    for (const std::string& key : checkpoint.alternatives) {
        if (failed == g_failedStrategicBranches.end() ||
            failed->second.find(key) == failed->second.end()) ++remaining;
    }
    return remaining;
}

static void failStrategicCheckpoint(DecisionCheckpoint& checkpoint) {
    checkpoint.failed = true;
    g_failedStrategicBranches[checkpoint.rootExactKey].insert(
        checkpoint.selectedMoveKey);
    discardPlannerRootEdge(checkpoint.rootExactKey,
                           checkpoint.selectedMoveKey);
    for (const std::string& key : checkpoint.pathStateKeys) {
        int& failures = g_strategicStateFailures[key];
        failures = std::min(5, failures + 1);
    }
    if (!checkpoint.selectedResultKey.empty()) {
        int& failures = g_strategicStateFailures[checkpoint.selectedResultKey];
        failures = std::min(5, failures + 1);
    }
}

static int strategicLinePenalty(const GameState& root,
                                const std::vector<Move>& line) {
    GameState cursor = root;
    int penalty = 0;
    const int limit = std::min(20, (int)line.size());
    for (int i = 0; i < limit; ++i) {
        if (!applyVirtualMove(cursor, line[i])) return 1500000;
        if (virtualFoundationCount(cursor) == 52) return 0;
        const int statePenalty = strategicStatePenalty(solverPositionKey(cursor));
        // Earlier re-entry into a failed structure is stronger evidence than
        // a distant overlap late in an otherwise distinct line.
        penalty += statePenalty / (1 + i / 4);
        if (penalty >= 1800000) return 1800000;
    }
    return penalty;
}

static bool moveCreatesIrreversibleProgress(const GameState& state,
                                            const Move& move) {
    GameState after = state;
    if (!applyVirtualMove(after, move)) return false;
    const int beforeFoundation = virtualFoundationCount(state);
    const int beforeDebt = foundationRiskDebt(state.foundations);
    const ProgressMetrics& epoch = g_progressTracker.baseline;
    return durableProgressFrom(
        after,
        g_progressTracker.initialized ? epoch.hidden : hiddenCardCount(state),
        g_progressTracker.initialized ? epoch.foundation : beforeFoundation,
        g_progressTracker.initialized ? epoch.foundationDebt : beforeDebt).any();
}

// A plan may contain several preparatory moves before it reaches the reveal or
// safe Foundation advance that made the branch attractive.  The old executor
// committed the complete searched line, including everything *after* that
// milestone.  That let a good search result drift into a later cosmetic
// Tableau shuffle without asking the solver to evaluate the freshly observed
// board again.  A certified prefix ends exactly at the first real milestone.
struct CertifiedPlanPrefix {
    bool valid = false;
    int progressIndex = -1;
    int failureIndex = -1;
    std::string reason;
    std::string failureStateKey;
};

static bool canonicalStateVisitedSinceProgress(const std::string& key) {
    if (!g_progressTracker.initialized) return false;
    const auto it = g_progressTracker.visitsSinceProgress.find(key);
    return it != g_progressTracker.visitsSinceProgress.end() && it->second > 0;
}

// When Talon is the only move, a repeated Draw-3 position is not necessarily
// useless: a later Waste exposure in the same bounded stock pass can have a
// legal exit to a Tableau or Foundation state not tried in this epoch. This
// proves only new reachability, not that the deal is won. Stop on a repeated
// stock state, so a full empty pass still cannot become an infinite loop.
static bool hasNovelTalonExit(
        const GameState& before, const GameState& afterDraw,
        const std::unordered_map<std::string, int>& visited,
        bool requireTalonOnlyAtRoot = true) {
    std::vector<Move> atRoot;
    generateVirtualMoves(before, atRoot);
    if (requireTalonOnlyAtRoot &&
        (atRoot.size() != 1 || atRoot.front().kind != VirtualMoveKind::Talon))
        return false;
    GameState cursor = afterDraw;
    std::unordered_set<std::string> passStates;
    const int cards = (int)before.talon.size() + (int)before.waste.size();
    const int draw = std::clamp(before.drawCount, 1, 3);
    const int maxSteps = (cards + draw - 1) / draw + 2;
    std::vector<Move> afterMoves;
    afterMoves.reserve(32);
    for (int step = 0; step < maxSteps; ++step) {
        const std::string stateKey = solverPositionKey(cursor);
        if (!passStates.insert(stateKey).second) break;
        generateVirtualMoves(cursor, afterMoves);
        for (const Move& followup : afterMoves) {
            if (followup.kind == VirtualMoveKind::Talon ||
                followup.kind == VirtualMoveKind::FoundationToTableau) continue;
            GameState exit = cursor;
            if (!applyVirtualMove(exit, followup)) continue;
            if (visited.find(solverPositionKey(exit)) == visited.end())
                return true;
        }
        if (!applyVirtualMove(cursor,
                              {VirtualMoveKind::Talon, -1, -1, -1})) break;
    }
    return false;
}

// Return a legal, strategically safe Waste -> Foundation action. Aces and
// Twos are unconditional unlockers; higher cards use the existing balanced-
// Foundation safety proof. `lowRanksOnly` is used while looking ahead through
// stock so the fast lane cannot chase a distant cosmetic push instead of a
// nearer Tableau reveal.
static bool findSafeWasteFoundationMove(const GameState& state, Move& selected,
                                        bool lowRanksOnly) {
    if (state.waste.empty()) return false;
    const Card& exposed = state.waste.back();
    const int rank = cardRank(exposed);
    if (rank < 0 || (lowRanksOnly && rank > 1)) return false;
    if (rank > 1 && !foundationAdvanceIsSafe(exposed, state.foundations))
        return false;
    for (int dst = 0; dst < 4; ++dst) {
        const Card* top = state.foundations[dst].empty()
            ? nullptr : &state.foundations[dst].back();
        if (!canDropOnFoundation(exposed, top)) continue;
        selected = {VirtualMoveKind::Foundation, -1, dst, -1};
        return true;
    }
    return false;
}

// Build the shortest exact Draw-N route to a useful low Foundation card in a
// single bounded stock orbit. The currently exposed Waste card is considered
// first, including every safe rank. Future exposures are restricted to Ace/
// Two, which fixes the captured "pass the Ace twice" behavior without letting
// a distant optional Foundation push pre-empt a stronger Tableau plan.
static bool findPriorityStockFoundationLine(const GameState& root,
                                             std::vector<Move>& line) {
    line.clear();
    Move foundation;
    if (findSafeWasteFoundationMove(root, foundation, false)) {
        line.push_back(foundation);
        return true;
    }
    const int cards = (int)root.talon.size() + (int)root.waste.size();
    if (cards <= 0) return false;
    const int drawCount = std::clamp(root.drawCount, 1, 3);
    const int maxSteps = (cards + drawCount - 1) / drawCount + 2;
    const Move draw{VirtualMoveKind::Talon, -1, -1, -1};
    GameState cursor = root;
    std::unordered_set<std::string> orbit;
    orbit.reserve((size_t)maxSteps + 1);
    line.reserve((size_t)maxSteps + 1);
    orbit.insert(solverPositionKey(cursor));
    for (int step = 0; step < maxSteps; ++step) {
        if (!applyVirtualMove(cursor, draw)) break;
        line.push_back(draw);
        if (cursor.recycles - root.recycles > 1) break;
        if (findSafeWasteFoundationMove(cursor, foundation, true)) {
            line.push_back(foundation);
            return true;
        }
        if (!orbit.insert(solverPositionKey(cursor)).second) break;
    }
    line.clear();
    return false;
}

static bool hasPriorityStockFoundationExit(const GameState& state) {
    std::vector<Move> line;
    return findPriorityStockFoundationLine(state, line) && !line.empty() &&
           line.front().kind == VirtualMoveKind::Talon;
}

// KING V17 stock router.  When Talon is the only legal root action, inspect the
// complete exact Draw-N orbit once and retain the best real exit (Waste ->
// Foundation/Tableau), not merely the next blind click.  Paid recycles are the
// primary comparison key because the fourth and later pass loses 20 real score
// points.  Within the same pass cost, prefer the strongest exit and then the
// shortest physical route.  A paid pass is intentionally not forbidden: some
// deals require one to remain solvable.
static bool findKingStockExitLine(const GameState& root,
                                  std::vector<Move>& line,
                                  int* paidRecyclesOut = nullptr) {
    line.clear();
    if (paidRecyclesOut) *paidRecyclesOut = 0;
    if (root.talon.empty() && root.waste.empty()) return false;

    std::vector<Move> rootMoves;
    rootMoves.reserve(32);
    generateVirtualMoves(root, rootMoves);
    if (rootMoves.size() != 1 ||
        rootMoves.front().kind != VirtualMoveKind::Talon) return false;

    const int cards = (int)root.talon.size() + (int)root.waste.size();
    const int drawCount = std::clamp(root.drawCount, 1, 3);
    const int maxSteps = 2 * ((cards + drawCount - 1) / drawCount) + 2;
    const int rootPaid = std::max(0, root.recycles - SOL_FREE_TALON_RECYCLES);
    const bool perfect = completeKnownDeck(root) && !hasOpaqueCards(root);
    const Move draw{VirtualMoveKind::Talon, -1, -1, -1};
    GameState cursor = root;
    std::unordered_set<std::string> orbit;
    orbit.reserve((size_t)maxSteps + 1);
    orbit.insert(solverPositionKey(cursor));
    std::vector<Move> draws;
    draws.reserve((size_t)maxSteps);
    std::vector<Move> exits;
    exits.reserve(32);

    int bestPaid = std::numeric_limits<int>::max();
    int bestQuality = std::numeric_limits<int>::min();
    int bestSteps = std::numeric_limits<int>::max();
    std::vector<Move> best;
    for (int step = 1; step <= maxSteps; ++step) {
        if (!applyVirtualMove(cursor, draw)) break;
        draws.push_back(draw);
        const int paid = std::max(0, cursor.recycles -
                                  SOL_FREE_TALON_RECYCLES) - rootPaid;
        if (paid > 1) break;

        generateVirtualMoves(cursor, exits);
        for (const Move& exit : exits) {
            if (exit.kind == VirtualMoveKind::Talon ||
                exit.kind == VirtualMoveKind::FoundationToTableau ||
                exit.from >= 0) continue;
            if (exit.kind == VirtualMoveKind::Foundation) {
                if (cursor.waste.empty()) continue;
                const Card& card = cursor.waste.back();
                if (cardRank(card) > 1 &&
                    !foundationAdvanceIsSafe(card, cursor.foundations)) continue;
            }
            const int tier = immediateMoveTier(cursor, exit);
            if (tier < 70) continue;
            GameState after = cursor;
            if (!applyVirtualMove(after, exit)) continue;
            if (perfect && provenDependencyDeadEnd(after)) continue;
            if (canonicalStateVisitedSinceProgress(solverPositionKey(after)))
                continue;

            int quality = tier * 260 - step * 700;
            if (exit.kind == VirtualMoveKind::Foundation &&
                !cursor.waste.empty() && cardRank(cursor.waste.back()) <= 1)
                quality += 4000;
            const int physicalSteps = step + 1;
            if (paid < bestPaid ||
                (paid == bestPaid && quality > bestQuality) ||
                (paid == bestPaid && quality == bestQuality &&
                 physicalSteps < bestSteps)) {
                bestPaid = paid;
                bestQuality = quality;
                bestSteps = physicalSteps;
                best = draws;
                best.push_back(exit);
            }
        }
        if (!orbit.insert(solverPositionKey(cursor)).second) break;
    }
    if (best.empty()) return false;
    line = std::move(best);
    if (paidRecyclesOut) *paidRecyclesOut = bestPaid;
    return true;
}

static CertifiedPlanPrefix certifyPlanPrefix(const GameState& root,
                                              const std::vector<Move>& line,
                                              bool allowRiskyFoundation,
                                              bool consultPhysicalHistory = true,
                                              bool allowRecoveryPreparation = false) {
    CertifiedPlanPrefix result;
    // Retained in the internal signature to keep call sites stable while the
    // former risky-root path is retired. Risk alone is never authorization to
    // execute; the searched line must prove one of the milestones below.
    (void)allowRiskyFoundation;
    if (line.empty()) {
        result.reason = "empty-plan";
        return result;
    }

    const int rootHidden = hiddenCardCount(root);
    const int rootFoundation = virtualFoundationCount(root);
    const int rootDebt = foundationRiskDebt(root.foundations);
    const bool rootAutoCompleteLayout = provenAutoCompleteLayout(root);
    const int rootMobility = virtualMobility(root);
    const int rootWasteAccess = usefulWasteAccessScore(root);
    const int rootDependency = dependencyPenalty(root);
    const int rootStructural = virtualStructuralScore(root);
    const ProgressMetrics& epoch = g_progressTracker.baseline;
    const int referenceHidden = consultPhysicalHistory && g_progressTracker.initialized ?
        epoch.hidden : rootHidden;
    const int referenceFoundation = consultPhysicalHistory && g_progressTracker.initialized ?
        epoch.foundation : rootFoundation;
    const int referenceDebt = consultPhysicalHistory && g_progressTracker.initialized ?
        epoch.foundationDebt : rootDebt;
    GameState cursor = root;
    std::unordered_set<std::string> localCanonicalStates;
    localCanonicalStates.insert(solverPositionKey(root));

    for (int i = 0; i < (int)line.size(); ++i) {
        if (!applyVirtualMove(cursor, line[i])) {
            result.failureIndex = i;
            result.reason = "illegal-virtual-step";
            return result;
        }

        const int foundationNow = virtualFoundationCount(cursor);
        const int debtNow = foundationRiskDebt(cursor.foundations);
        const DurableProgressEvidence durable = durableProgressFrom(
            cursor, referenceHidden, referenceFoundation, referenceDebt);
        const bool autoCompleteLayout = !rootAutoCompleteLayout &&
            provenAutoCompleteLayout(cursor);
        // Pulling a card back from Foundation can lower the debt metric, but it
        // is a reversible bridge, not progress. Otherwise the progress epoch
        // advances and the same card can ping-pong back immediately.
        // recordRootPlan() deliberately retains a searched bridge move during
        // strategic recovery when it improves mobility, Waste accessibility,
        // dependency pressure, or the global structure without adding
        // Foundation debt. The executor previously had no matching definition,
        // so every one of those retained roots was later rejected as
        // "no-proven-progress" and the live board appeared frozen. Accept the
        // first matching bridge only in Recovery mode; normal play still
        // requires an irreversible reveal/Foundation milestone.
        const bool recoveryPreparation = allowRecoveryPreparation && i >= 1 &&
            !durable.any() &&
            debtNow <= rootDebt && foundationNow <= rootFoundation &&
            (virtualMobility(cursor) > rootMobility ||
             usefulWasteAccessScore(cursor) > rootWasteAccess ||
             rootDependency - dependencyPenalty(cursor) >= 120 ||
             virtualStructuralScore(cursor) - rootStructural >= 900);
        const bool reachedMilestone = durable.any() || autoCompleteLayout ||
                                      recoveryPreparation;
        const std::string canonicalKey = solverPositionKey(cursor);

        // Column indices are presentation details. Re-entering the same
        // canonical structure before making progress is the exact signature
        // of King/sequence ping-pong, even when the physical column numbers
        // differ and virtualPositionKey therefore looks new.
        if (!durable.any() &&
            localCanonicalStates.find(canonicalKey) != localCanonicalStates.end()) {
            result.failureIndex = i;
            result.reason = "canonical-cycle-in-plan";
            result.failureStateKey = canonicalKey;
            return result;
        }
        // The physical dispatcher has one narrow Draw-3 bridge exception:
        // a repeated stock position may be crossed once when a bounded pass
        // exposes a genuinely new card move.  Certification must agree with
        // that rule.  Otherwise it rejects every searched Talon-first line at
        // step zero while the search keeps promoting another backup forever.
        const bool visitedPhysicalState = !durable.any() &&
            consultPhysicalHistory &&
            canonicalStateVisitedSinceProgress(canonicalKey);
        const bool permittedRootTalonBridge = visitedPhysicalState && i == 0 &&
            line[i].kind == VirtualMoveKind::Talon &&
            physicalCandidateAllowed(root, line[i], nowMs());
        if (visitedPhysicalState && !permittedRootTalonBridge) {
            result.failureIndex = i;
            result.reason = "visited-no-progress-state";
            result.failureStateKey = canonicalKey;
            return result;
        }
        localCanonicalStates.insert(canonicalKey);

        if (reachedMilestone) {
            result.valid = true;
            result.progressIndex = i;
            result.reason = durable.won ? "win" :
                            (autoCompleteLayout ? "auto-complete-layout" :
                            (durable.revealed ? "hidden-reveal" :
                             (durable.repaidDebt ? "foundation-debt-reduced" :
                              (durable.safeFoundation ? "safe-foundation" :
                                                    "recovery-preparation"))));
            return result;
        }
    }

    result.failureIndex = (int)line.size();
    result.reason = "no-proven-progress";
    result.failureStateKey = solverPositionKey(cursor);
    return result;
}

// An immediately exposed hidden card outranks a merely playable Waste card.
// Keep this selection separate from dispatch so the priority can be tested
// without calling into Unity or changing the existing move/ABI interfaces.
static bool findEmergencyTableauReveal(const GameState& state, Move& selected) {
    for (int src = 0; src < 7; ++src) {
        const int first = firstFaceUpIndex(state.tableau[src]);
        if (first <= 0 || state.tableau[src][first - 1].faceUp()) continue;
        const Card& card = state.tableau[src][first];
        for (int dst = 0; dst < 7; ++dst) {
            if (!isStrategicTableauMove(state.tableau, src, first, dst, card)) continue;
            const Move move{VirtualMoveKind::Tableau, src, dst, first};
            if (!physicalCandidateAllowed(state, move, nowMs())) continue;
            selected = move;
            return true;
        }
    }
    return false;
}

static bool runPlanGuardSelfTests() {
    GameState knowledgeBefore;
    Card opaqueForAck;
    opaqueForAck.id = 7;
    knowledgeBefore.tableau[0].push_back(opaqueForAck);
    GameState knowledgeAfter = knowledgeBefore;
    knowledgeAfter.tableau[0].back().knownValue = 25;
    const bool knowledgeIsNotPhysical =
        virtualPositionKey(knowledgeBefore) != virtualPositionKey(knowledgeAfter) &&
        physicalPositionKey(knowledgeBefore) == physicalPositionKey(knowledgeAfter);
    GameState expectedReveal = knowledgeBefore;
    expectedReveal.tableau[0].back().virtuallyRevealed = true;
    GameState observedReveal = knowledgeAfter;
    observedReveal.tableau[0].back().value = 25;
    const bool revealAckMatches =
        physicalPositionKey(expectedReveal) == physicalPositionKey(observedReveal) &&
        physicalPositionKey(knowledgeBefore) != physicalPositionKey(observedReveal);
    GameState opaqueStock;
    Card opaqueDraw;
    opaqueDraw.id = 8;
    opaqueStock.talon.push_back(opaqueDraw);
    GameState expectedDraw = opaqueStock;
    const Move opaqueDrawMove{VirtualMoveKind::Talon, -1, -1, -1};
    const bool simulatedOpaqueDraw = applyVirtualMove(expectedDraw,
                                                      opaqueDrawMove);
    GameState observedDraw = expectedDraw;
    if (!observedDraw.waste.empty()) {
        observedDraw.waste.back().value = 25;
        observedDraw.waste.back().knownValue = 25;
    }
    const bool opaqueDrawAckMatches = simulatedOpaqueDraw &&
        physicalPositionKey(expectedDraw) == physicalPositionKey(observedDraw) &&
        physicalPositionKey(opaqueStock) != physicalPositionKey(observedDraw);

    GameState delayedKing;
    Card clearSource;
    clearSource.id = 4;
    clearSource.value = 4; // black five, can move onto a red six.
    clearSource.knownValue = 4;
    Card clearTarget;
    clearTarget.id = 18;
    clearTarget.value = 18;
    clearTarget.knownValue = 18;
    delayedKing.tableau[0].push_back(clearSource);
    delayedKing.tableau[1].push_back(clearTarget);
    constexpr int stockValues[] = {0, 1, 2, 3, 5, 6, 7, 8, 11};
    for (int i = 0; i < 9; ++i) {
        Card stockCard;
        stockCard.id = 20 + i;
        stockCard.value = stockValues[i]; // King exposed on third Draw-3.
        stockCard.knownValue = stockCard.value;
        delayedKing.talon.push_back(stockCard);
    }
    std::vector<Move> normalClears;
    std::vector<Move> recoveryClears;
    generateVirtualMoves(delayedKing, normalClears);
    generateVirtualMoves(delayedKing, recoveryClears, false, true);
    const auto containsClear = [](const std::vector<Move>& moves) {
        return std::any_of(moves.begin(), moves.end(), [](const Move& move) {
            return move.kind == VirtualMoveKind::Tableau &&
                   move.from == 0 && move.to == 1 && move.start == 0;
        });
    };
    const bool recoveryKeepsDistantKingPath =
        !containsClear(normalClears) && containsClear(recoveryClears);

    Card king;
    king.id = 11;
    king.value = 11; // Jawaker value 11 maps to King (rank 12).
    king.knownValue = 11;
    GameState left;
    left.tableau[0].push_back(king);
    GameState right = left;
    right.tableau[0].clear();
    right.tableau[5].push_back(king);
    const bool canonicalColumns = solverPositionKey(left) == solverPositionKey(right) &&
                                  virtualPositionKey(left) != virtualPositionKey(right);
    const bool compactKeyKeepsKnowledge =
        solverPositionKey(knowledgeBefore) != solverPositionKey(knowledgeAfter);
    GameState keyStockA;
    Card stockA;
    stockA.id = 20;
    stockA.value = stockA.knownValue = 0;
    Card stockB;
    stockB.id = 21;
    stockB.value = stockB.knownValue = 1;
    keyStockA.talon = {stockA, stockB};
    GameState keyStockB = keyStockA;
    std::swap(keyStockB.talon[0], keyStockB.talon[1]);
    const bool compactKeyKeepsStockOrder =
        solverPositionKey(keyStockA) != solverPositionKey(keyStockB);
    GameState syntheticIdA;
    Card synthetic;
    synthetic.id = 200;
    synthetic.value = synthetic.knownValue = 3;
    syntheticIdA.tableau[0].push_back(synthetic);
    GameState syntheticIdB = syntheticIdA;
    syntheticIdB.tableau[0].back().id = 456;
    const bool compactKeyKeepsWideIds =
        solverPositionKey(syntheticIdA) != solverPositionKey(syntheticIdB);

    // A full Draw-3 followed by an immediate recycle returns to the same
    // structural state and must be rejected as a no-progress cycle.
    GameState recycle;
    recycle.drawCount = 3;
    for (int i = 0; i < 3; ++i) {
        Card card;
        card.id = 20 + i;
        card.value = i;
        card.knownValue = i;
        recycle.talon.push_back(card);
    }
    const std::vector<Move> recycleLine{
        {VirtualMoveKind::Talon, -1, -1, -1},
        {VirtualMoveKind::Talon, -1, -1, -1}
    };
    GameState drawn = recycle;
    const bool drawChangesPhysical =
        applyVirtualMove(drawn, recycleLine.front()) &&
        physicalPositionKey(drawn) != physicalPositionKey(recycle);
    const CertifiedPlanPrefix cycle = certifyPlanPrefix(
        recycle, recycleLine, false, false);
    const bool rejectsCycle = !cycle.valid &&
                              cycle.reason == "canonical-cycle-in-plan";
    const bool drawPreservesNativeOrder = drawn.talon.empty() &&
        drawn.waste.size() == 3 && drawn.waste[0].id == 20 &&
        drawn.waste[1].id == 21 && drawn.waste[2].id == 22;

    // A canonical repeat can be crossed once if the exposed Waste card opens
    // a genuinely new non-Talon state. If that exit was also visited, or if
    // the drawn card cannot move, the original no-loop guard must still win.
    GameState talonBridge;
    Card bridgeTarget;
    bridgeTarget.id = 40;
    bridgeTarget.value = bridgeTarget.knownValue = 18; // red six
    talonBridge.tableau[0].push_back(bridgeTarget);
    Card bridgeCard;
    bridgeCard.id = 41;
    bridgeCard.value = bridgeCard.knownValue = 4; // black five
    talonBridge.talon.push_back(bridgeCard);
    GameState bridgeDrawn = talonBridge;
    const bool bridgeDrawnOkay = applyVirtualMove(bridgeDrawn, opaqueDrawMove);
    std::unordered_map<std::string, int> bridgeVisits;
    bridgeVisits.emplace(solverPositionKey(bridgeDrawn), 1);
    const bool opensNovelTalonExit = bridgeDrawnOkay &&
        hasNovelTalonExit(talonBridge, bridgeDrawn, bridgeVisits);
    GameState bridgeExited = bridgeDrawn;
    const bool bridgeExitApplied = applyVirtualMove(
        bridgeExited, {VirtualMoveKind::Tableau, -1, 0, -1});
    bridgeVisits.emplace(solverPositionKey(bridgeExited), 1);
    const bool rejectsVisitedTalonExit = bridgeExitApplied &&
        !hasNovelTalonExit(talonBridge, bridgeDrawn, bridgeVisits);
    GameState noExitBridge = talonBridge;
    noExitBridge.talon.back().value = noExitBridge.talon.back().knownValue = 3;
    GameState noExitDrawn = noExitBridge;
    const bool rejectsTalonWithoutExit =
        applyVirtualMove(noExitDrawn, opaqueDrawMove) &&
        !hasNovelTalonExit(noExitBridge, noExitDrawn, bridgeVisits);
    GameState laterBridge = talonBridge;
    laterBridge.talon.clear();
    for (int id = 42; id < 45; ++id) {
        Card filler;
        filler.id = id;
        filler.value = filler.knownValue = 3; // no exit on first Draw-3
        laterBridge.talon.push_back(filler);
    }
    bridgeCard.id = 45;
    laterBridge.talon.push_back(bridgeCard); // exposed by the next draw
    GameState laterDrawn = laterBridge;
    const bool firstLaterDrawOkay = applyVirtualMove(laterDrawn, opaqueDrawMove);
    std::unordered_map<std::string, int> laterVisits;
    laterVisits.emplace(solverPositionKey(laterDrawn), 1);
    const bool opensLaterTalonExit = firstLaterDrawOkay &&
        hasNovelTalonExit(laterBridge, laterDrawn, laterVisits);

    // Liveness must also see a useful stock exit when another legal Waste move
    // exists at the current root. The ordinary one-move bridge remains strict;
    // only the post-timeout proof opts into this wider root scan.
    GameState multiMoveLiveness = talonBridge;
    Card redEight;
    redEight.id = 52;
    redEight.value = redEight.knownValue = 20;
    multiMoveLiveness.tableau[1].push_back(redEight);
    Card blackSeven;
    blackSeven.id = 53;
    blackSeven.value = blackSeven.knownValue = 6;
    multiMoveLiveness.waste.push_back(blackSeven);
    GameState multiMoveDrawn = multiMoveLiveness;
    std::unordered_map<std::string, int> livenessVisits;
    const bool multiMoveDrawApplied =
        applyVirtualMove(multiMoveDrawn, opaqueDrawMove);
    const bool livenessFindsNovelExit = multiMoveDrawApplied &&
        !hasNovelTalonExit(multiMoveLiveness, multiMoveDrawn,
                           livenessVisits) &&
        hasNovelTalonExit(multiMoveLiveness, multiMoveDrawn,
                          livenessVisits, false);

    // Regression for the captured 601-point freeze: the dispatcher accepted a
    // one-time stock bridge, but the planner's prefix guard rejected the same
    // first step as already visited and searched hundreds of thousands of
    // alternatives without submitting a move.
    GameState certifiedBridge;
    Card bridgeAce;
    bridgeAce.id = 46;
    bridgeAce.value = bridgeAce.knownValue = 12;
    certifiedBridge.talon.push_back(bridgeAce);
    GameState certifiedDrawn = certifiedBridge;
    const bool certifiedDrawnOkay = applyVirtualMove(certifiedDrawn, opaqueDrawMove);
    const StrategicProgressTracker savedTracker = g_progressTracker;
    const auto savedBridgeAttempts = g_talonBridgeAttempted;
    g_progressTracker.initialized = true;
    g_progressTracker.movesWithoutProgress = 2;
    g_progressTracker.visitsSinceProgress.clear();
    if (certifiedDrawnOkay)
        g_progressTracker.visitsSinceProgress.emplace(
            solverPositionKey(certifiedDrawn), 1);
    g_talonBridgeAttempted.clear();
    const std::vector<Move> bridgeToFoundation{
        opaqueDrawMove,
        {VirtualMoveKind::Foundation, -1, 0, -1}
    };
    const CertifiedPlanPrefix allowedBridge = certifyPlanPrefix(
        certifiedBridge, bridgeToFoundation, false, true);
    const bool certifiesNovelTalonBridge = certifiedDrawnOkay &&
        allowedBridge.valid && allowedBridge.progressIndex == 1;
    g_talonBridgeAttempted.insert(virtualPositionKey(certifiedBridge));
    const CertifiedPlanPrefix blockedBridge = certifyPlanPrefix(
        certifiedBridge, bridgeToFoundation, false, true);
    const bool rejectsRepeatedTalonBridge = !blockedBridge.valid &&
        blockedBridge.reason == "visited-no-progress-state" &&
        blockedBridge.failureIndex == 0;
    g_progressTracker = savedTracker;
    g_talonBridgeAttempted = savedBridgeAttempts;

    // Moving the exposed sequence below reveals a known face-down card.  This
    // is genuine progress and the certified prefix must end at that first move.
    GameState reveal;
    Card hidden;
    hidden.id = 31;
    hidden.value = -1;
    hidden.knownValue = 0;
    Card moving;
    moving.id = 32;
    moving.value = 4;  // rank 5, black
    moving.knownValue = 4;
    Card destination;
    destination.id = 33;
    destination.value = 18; // rank 6, red
    destination.knownValue = 18;
    reveal.tableau[0] = {hidden, moving};
    reveal.tableau[1] = {destination};
    const std::vector<Move> revealLine{
        {VirtualMoveKind::Tableau, 0, 1, 1}
    };
    const CertifiedPlanPrefix progress = certifyPlanPrefix(
        reveal, revealLine, false, false);
    const bool acceptsReveal = progress.valid && progress.progressIndex == 0;
    GameState movedSequence = reveal;
    const bool tableauMoveKeepsReveal =
        applyVirtualMove(movedSequence, revealLine.front()) &&
        movedSequence.tableau[0].size() == 1 &&
        movedSequence.tableau[0].back().faceUp() &&
        movedSequence.tableau[1].size() == 2 &&
        movedSequence.tableau[1].back().id == moving.id;
    GameState opaqueReveal = reveal;
    opaqueReveal.tableau[0][0].knownValue = -1;
    GameState opaqueRevealed = opaqueReveal;
    const bool scoresUnmappedReveal =
        applyVirtualMove(opaqueRevealed, revealLine.front()) &&
        opaqueRevealed.tableau[0].back().virtuallyRevealed &&
        candidateTacticalBonus(opaqueReveal, opaqueRevealed,
                               revealLine.front()) >= 12000;

    // A legal Waste -> Tableau move used to run before the one-step hidden
    // reveal below, even though it reveals nothing. The direct fast path must
    // find the reveal; the Waste move is left for bounded searched ranking.
    GameState emergencyChoice = reveal;
    Card wasteFive;
    wasteFive.id = 34;
    wasteFive.value = wasteFive.knownValue = 30; // other black five
    emergencyChoice.waste.push_back(wasteFive);
    const Move legalWasteMove{VirtualMoveKind::Tableau, -1, 1, -1};
    GameState afterWaste = emergencyChoice;
    GameState afterReveal = emergencyChoice;
    Move preferredReveal;
    const bool revealBeforeWaste =
        applyVirtualMove(afterWaste, legalWasteMove) &&
        applyVirtualMove(afterReveal, revealLine.front()) &&
        findEmergencyTableauReveal(emergencyChoice, preferredReveal) &&
        preferredReveal.kind == VirtualMoveKind::Tableau &&
        preferredReveal.from == 0 && preferredReveal.to == 1 &&
        preferredReveal.start == 1 &&
        hiddenCardCount(afterReveal) < hiddenCardCount(emergencyChoice) &&
        hiddenCardCount(afterWaste) == hiddenCardCount(emergencyChoice) &&
        candidateTacticalBonus(emergencyChoice, afterReveal,
                               preferredReveal) >
            candidateTacticalBonus(emergencyChoice, afterWaste,
                                   legalWasteMove);

    // A debt-only Foundation push used to be accepted by the risky fallback,
    // reset the no-progress epoch, and let one suit race ahead. Keep searching
    // through it, but never certify the one-step push as an executable plan.
    GameState debtOnly;
    Card ace;
    ace.id = 40;
    ace.value = 12; // Ace of suit 0.
    ace.knownValue = 12;
    Card two;
    two.id = 41;
    two.value = 0; // Two of suit 0.
    two.knownValue = 0;
    Card three;
    three.id = 42;
    three.value = 1; // Three of suit 0; unsafe while opposite suits are empty.
    three.knownValue = 1;
    debtOnly.foundations[0] = {ace, two};
    debtOnly.tableau[0] = {three};
    const std::vector<Move> debtOnlyLine{
        {VirtualMoveKind::Foundation, 0, 0, -1}
    };
    const CertifiedPlanPrefix debtPush = certifyPlanPrefix(
        debtOnly, debtOnlyLine, true, false);
    const bool rejectsDebtOnly = !debtPush.valid &&
                                  debtPush.reason == "no-proven-progress" &&
                                  !moveCreatesIrreversibleProgress(debtOnly,
                                                                   debtOnlyLine.front());

    // A locally safe Foundation re-add after a pullback merely recovers the
    // epoch's old high-water mark. It must not reset cycle history or certify
    // a new root; a continuation that exceeds that mark must remain legal.
    GameState foundationHighWater;
    Card firstAce;
    firstAce.id = 47;
    firstAce.value = firstAce.knownValue = 12;
    foundationHighWater.foundations[0].push_back(firstAce);
    GameState foundationRebuild = foundationHighWater;
    foundationRebuild.foundations[0].clear();
    foundationRebuild.tableau[0].push_back(firstAce);
    const Move readdAce{VirtualMoveKind::Foundation, 0, 0, -1};
    GameState rebuiltFoundation = foundationRebuild;
    const bool rebuiltAceApplied = applyVirtualMove(rebuiltFoundation, readdAce);
    const StrategicProgressTracker savedHighWaterTracker = g_progressTracker;
    g_progressTracker = StrategicProgressTracker{};
    g_progressTracker.initialized = true;
    g_progressTracker.baseline = collectProgressMetrics(foundationHighWater, 0);
    if (rebuiltAceApplied)
        g_progressTracker.visitsSinceProgress[solverPositionKey(rebuiltFoundation)] = 1;
    const CertifiedPlanPrefix repeatedFoundation = certifyPlanPrefix(
        foundationRebuild, {readdAce}, false, true);
    const bool rejectsFoundationRebuild = rebuiltAceApplied &&
        !moveCreatesIrreversibleProgress(foundationRebuild, readdAce) &&
        !repeatedFoundation.valid &&
        repeatedFoundation.reason == "visited-no-progress-state" &&
        !physicalCandidateAllowed(foundationRebuild, readdAce, nowMs());

    GameState foundationBridge = foundationRebuild;
    Card secondAce;
    secondAce.id = 48;
    secondAce.value = secondAce.knownValue = 25;
    foundationBridge.talon.push_back(secondAce);
    const std::vector<Move> bridgePastHighWater{
        readdAce,
        {VirtualMoveKind::Talon, -1, -1, -1},
        {VirtualMoveKind::Foundation, -1, 1, -1}
    };
    g_progressTracker.visitsSinceProgress.clear();
    const CertifiedPlanPrefix exceededFoundation = certifyPlanPrefix(
        foundationBridge, bridgePastHighWater, false, true);
    const bool allowsFoundationAboveHighWater = exceededFoundation.valid &&
        exceededFoundation.progressIndex == 2 &&
        exceededFoundation.reason == "safe-foundation";

    // With two Kings in Draw-1 stock, a second draw makes empty columns look
    // much more mobile, but no card is revealed and no Foundation advances.
    // Mark the structural gain as visited: recovery must not call that state a
    // progress milestone and bypass the canonical-history guard.
    GameState reversibleDraws;
    reversibleDraws.drawCount = 1;
    Card kingA;
    kingA.id = 49;
    kingA.value = kingA.knownValue = 11;
    Card kingB;
    kingB.id = 50;
    kingB.value = kingB.knownValue = 24;
    reversibleDraws.talon = {kingA, kingB};
    const Move drawOne{VirtualMoveKind::Talon, -1, -1, -1};
    const std::vector<Move> reversibleDrawLine{drawOne, drawOne};
    GameState afterReversibleDraws = reversibleDraws;
    const bool bothDrawsApplied =
        applyVirtualMove(afterReversibleDraws, drawOne) &&
        applyVirtualMove(afterReversibleDraws, drawOne);
    g_progressTracker.baseline = collectProgressMetrics(reversibleDraws, 0);
    g_progressTracker.visitsSinceProgress.clear();
    const CertifiedPlanPrefix novelPreparation = certifyPlanPrefix(
        reversibleDraws, reversibleDrawLine, false, true, true);
    if (bothDrawsApplied)
        g_progressTracker.visitsSinceProgress[solverPositionKey(afterReversibleDraws)] = 1;
    const CertifiedPlanPrefix repeatedPreparation = certifyPlanPrefix(
        reversibleDraws, reversibleDrawLine, false, true, true);
    const bool rejectsVisitedRecoveryPreparation = bothDrawsApplied &&
        novelPreparation.valid &&
        novelPreparation.reason == "recovery-preparation" &&
        !repeatedPreparation.valid &&
        repeatedPreparation.reason == "visited-no-progress-state";
    g_progressTracker = savedHighWaterTracker;

    // With four values unresolved, reducing the number of physical blockers
    // above an unknown must improve the pessimistic remaining-value bound.
    // This test uses all 52 unique values except the four assigned to opaque
    // physical cards, matching the captured 48/52 GameStore condition.
    GameState uncertaintyBuried;
    const std::array<int, 4> omittedValues{0, 11, 12, 25};
    for (int i = 0; i < 4; ++i) {
        Card opaque;
        opaque.id = 100 + i;
        uncertaintyBuried.tableau[i].push_back(opaque);
    }
    for (int value = 0; value < 52; ++value) {
        if (std::find(omittedValues.begin(), omittedValues.end(), value) !=
            omittedValues.end()) continue;
        Card known;
        known.id = 200 + value;
        known.value = value;
        known.knownValue = value;
        uncertaintyBuried.talon.push_back(known);
    }
    // Put three known cards above the first opaque card. Moving one away must
    // lower the worst-case dependency risk without knowing the hidden value.
    for (int i = 0; i < 3; ++i) {
        uncertaintyBuried.tableau[0].push_back(uncertaintyBuried.talon.back());
        uncertaintyBuried.talon.pop_back();
    }
    GameState uncertaintyShallower = uncertaintyBuried;
    uncertaintyShallower.waste.push_back(uncertaintyShallower.tableau[0].back());
    uncertaintyShallower.tableau[0].pop_back();
    const int buriedRisk = unresolvedValueDependencyPenalty(uncertaintyBuried);
    const int shallowerRisk = unresolvedValueDependencyPenalty(uncertaintyShallower);
    const bool ranksUncertainty = buriedRisk > shallowerRisk && shallowerRisk > 0;

    // A 51-card Foundation state with the last King in stock has an exact
    // two-step finish. The detector must prove Draw-N -> Foundation, and it
    // must refuse a line containing any extra action after the win.
    GameState finish;
    finish.drawCount = 3;
    for (int suit = 0; suit < 4; ++suit) {
        for (int order = 0; order < 13; ++order) {
            const int value = suit * 13 + (order + 12) % 13;
            Card card;
            card.id = value;
            card.value = value;
            card.knownValue = value;
            if (suit == 0 && order == 12) finish.talon.push_back(card);
            else finish.foundations[suit].push_back(card);
        }
    }
    std::vector<Move> finishLine;
    const bool findsExactFinish =
        findFoundationOnlyFinishPlan(finish, finishLine) &&
        finishLine.size() == 2 &&
        finishLine[0].kind == VirtualMoveKind::Talon &&
        finishLine[1].kind == VirtualMoveKind::Foundation &&
        verifiedWinningLine(finish, finishLine);
    FoundationFinishProbe resumedFinish;
    resumedFinish.cursor = finish;
    const bool resumesFinishProof =
        advanceFoundationOnlyFinishPlan(resumedFinish, 1, 0) ==
            FoundationFinishResult::Pending &&
        advanceFoundationOnlyFinishPlan(resumedFinish, 1, 0) ==
            FoundationFinishResult::Won &&
        verifiedWinningLine(finish, resumedFinish.line);
    finishLine.push_back({VirtualMoveKind::Talon, -1, -1, -1});
    const bool rejectsAfterWin = !verifiedWinningLine(finish, finishLine);

    GameState blockedFinish = finish;
    blockedFinish.tableau[0].push_back(blockedFinish.talon.back());
    blockedFinish.talon.clear();
    // Conceal the required final card behind an unrelated top card without
    // changing the complete deck. This cannot be called an automatic finish.
    Card blocker = blockedFinish.foundations[1].back();
    blockedFinish.foundations[1].pop_back();
    Card prerequisite = blockedFinish.foundations[1].back();
    blockedFinish.foundations[1].pop_back();
    blockedFinish.tableau[0].push_back(prerequisite);
    blockedFinish.tableau[0].push_back(blocker);
    std::vector<Move> blockedLine;
    const bool rejectsBlockedFinish =
        !findFoundationOnlyFinishPlan(blockedFinish, blockedLine);

    GameState autoLayout = finish;
    autoLayout.tableau[0].push_back(autoLayout.talon.back());
    autoLayout.talon.clear();
    std::vector<Move> directAutoLine;
    const bool recognizesNativeAutoLayout =
        virtualFoundationCount(autoLayout) == 51 &&
        autoCompleteLayoutCandidate(autoLayout) &&
        provenAutoCompleteLayout(autoLayout, &directAutoLine) &&
        directAutoLine.size() == 1 &&
        verifiedWinningLine(autoLayout, directAutoLine) &&
        virtualStructuralScore(autoLayout, 1) ==
            virtualStructuralScore(autoLayout);
    GameState wasteStillPresent = autoLayout;
    wasteStillPresent.waste.push_back(wasteStillPresent.tableau[0].back());
    wasteStillPresent.tableau[0].pop_back();
    GameState hiddenStillPresent = autoLayout;
    hiddenStillPresent.tableau[0].back().value = -1;
    hiddenStillPresent.tableau[0].back().virtuallyRevealed = false;
    const bool rejectsIncompleteAutoLayout =
        !autoCompleteLayoutCandidate(finish) &&
        !autoCompleteLayoutCandidate(wasteStillPresent) &&
        !autoCompleteLayoutCandidate(hiddenStillPresent) &&
        autoCompleteLayoutCandidate(blockedFinish) &&
        !provenAutoCompleteLayout(blockedFinish, &directAutoLine) &&
        directAutoLine.empty() &&
        virtualStructuralScore(blockedFinish, 0) ==
            virtualStructuralScore(blockedFinish);

    // Recycling is charged across the whole match, while the no-progress
    // recycle counter resets after a real card move. Verify that virtual
    // search keeps those two facts separate and prefers an otherwise equal
    // free pass without forbidding a charged pass needed for completion.
    GameState paidPass;
    paidPass.drawCount = 3;
    paidPass.recycles = SOL_FREE_TALON_RECYCLES;
    paidPass.stagnantRecycles = 2;
    Card aceForPass;
    aceForPass.id = 0;
    aceForPass.value = 12;
    paidPass.waste.push_back(aceForPass);
    GameState afterPaidPass = paidPass;
    const Move recycleMove{VirtualMoveKind::Talon, -1, -1, -1};
    const bool paidPassApplied = applyVirtualMove(afterPaidPass, recycleMove);
    GameState freePass = paidPass;
    freePass.recycles = SOL_FREE_TALON_RECYCLES - 1;
    GameState afterFreePass = freePass;
    const bool freePassApplied = applyVirtualMove(afterFreePass, recycleMove);
    const bool chargedPassRankedLower = paidPassApplied && freePassApplied &&
        candidateTacticalBonus(paidPass, afterPaidPass, recycleMove) <
        candidateTacticalBonus(freePass, afterFreePass, recycleMove);
    const bool accountPersistsAcrossProgress = paidPassApplied &&
        applyVirtualMove(afterPaidPass, recycleMove) &&
        applyVirtualMove(afterPaidPass,
                         {VirtualMoveKind::Foundation, -1, 0, -1}) &&
        afterPaidPass.recycles == SOL_FREE_TALON_RECYCLES + 1 &&
        afterPaidPass.stagnantRecycles == 0;
    const bool recycleAccounting = chargedPassRankedLower &&
        accountPersistsAcrossProgress;

    // Regression for the reported Draw-3 tempo loss: the Ace is exposed by
    // the second click and must be routed immediately, without recycling the
    // stock or waiting for the general search proof gate.
    GameState fastStock;
    fastStock.drawCount = 3;
    constexpr int fastStockValues[] = {5, 6, 7, 8, 9, 12};
    for (int i = 0; i < 6; ++i) {
        Card card;
        card.id = 500 + i;
        card.value = card.knownValue = fastStockValues[i];
        fastStock.talon.push_back(card);
    }
    std::vector<Move> fastStockLine;
    const bool fastStockFound =
        findPriorityStockFoundationLine(fastStock, fastStockLine);
    GameState fastStockResult = fastStock;
    bool fastStockApplied = fastStockFound;
    for (const Move& move : fastStockLine)
        fastStockApplied = fastStockApplied &&
                           applyVirtualMove(fastStockResult, move);
    const bool catchesAceOnFirstOrbit = fastStockApplied &&
        fastStockLine.size() == 3 &&
        fastStockLine[0].kind == VirtualMoveKind::Talon &&
        fastStockLine[1].kind == VirtualMoveKind::Talon &&
        fastStockLine[2].kind == VirtualMoveKind::Foundation &&
        fastStockResult.recycles == 0 &&
        virtualFoundationCount(fastStockResult) == 1;
    GameState exposedAce;
    exposedAce.waste.push_back(fastStock.talon.back());
    std::vector<Move> exposedAceLine;
    const bool playsExposedAceImmediately =
        findPriorityStockFoundationLine(exposedAce, exposedAceLine) &&
        exposedAceLine.size() == 1 &&
        exposedAceLine.front().kind == VirtualMoveKind::Foundation;
    std::vector<Move> kingStockLine;
    int kingPaidRecycles = -1;
    const bool kingRoutesUsefulExit =
        findKingStockExitLine(fastStock, kingStockLine,
                              &kingPaidRecycles) &&
        !kingStockLine.empty() &&
        kingStockLine.back().kind == VirtualMoveKind::Foundation &&
        kingPaidRecycles == 0;
    SOL_LOG("[KING_TEST] stockRoute=%d steps=%d paidRecycles=%d",
            kingRoutesUsefulExit ? 1 : 0, (int)kingStockLine.size(),
            kingPaidRecycles);

    SOL_LOG("[PLAN_GUARD_TEST] physicalKnowledge=%d revealAck=%d opaqueDrawAck=%d drawAck=%d distantKing=%d canonical=%d compact=%d/%d/%d drawOrder=%d talonBridge=%d/%d/%d/%d liveness=%d certificate=%d/%d tableau=%d opaqueReveal=%d revealBeforeWaste=%d cycle=%d reveal=%d debtOnly=%d highWater=%d/%d recoveryVisit=%d uncertainty=%d finish=%d resume=%d afterWin=%d blocked=%d autoLayout=%d/%d recycle=%d stockFast=%d/%d risk=%d>%d",
            knowledgeIsNotPhysical ? 1 : 0, revealAckMatches ? 1 : 0,
            opaqueDrawAckMatches ? 1 : 0,
            drawChangesPhysical ? 1 : 0,
            recoveryKeepsDistantKingPath ? 1 : 0,
            canonicalColumns ? 1 : 0,
            compactKeyKeepsKnowledge ? 1 : 0,
            compactKeyKeepsStockOrder ? 1 : 0,
            compactKeyKeepsWideIds ? 1 : 0,
            drawPreservesNativeOrder ? 1 : 0,
            opensNovelTalonExit ? 1 : 0,
            rejectsVisitedTalonExit ? 1 : 0,
            rejectsTalonWithoutExit ? 1 : 0,
            opensLaterTalonExit ? 1 : 0,
            livenessFindsNovelExit ? 1 : 0,
            certifiesNovelTalonBridge ? 1 : 0,
            rejectsRepeatedTalonBridge ? 1 : 0,
            tableauMoveKeepsReveal ? 1 : 0,
            scoresUnmappedReveal ? 1 : 0,
            revealBeforeWaste ? 1 : 0,
            rejectsCycle ? 1 : 0,
             acceptsReveal ? 1 : 0, rejectsDebtOnly ? 1 : 0,
             rejectsFoundationRebuild ? 1 : 0,
             allowsFoundationAboveHighWater ? 1 : 0,
             rejectsVisitedRecoveryPreparation ? 1 : 0,
            ranksUncertainty ? 1 : 0, findsExactFinish ? 1 : 0,
            resumesFinishProof ? 1 : 0,
            rejectsAfterWin ? 1 : 0, rejectsBlockedFinish ? 1 : 0,
            recognizesNativeAutoLayout ? 1 : 0,
            rejectsIncompleteAutoLayout ? 1 : 0,
            recycleAccounting ? 1 : 0,
            catchesAceOnFirstOrbit ? 1 : 0,
            playsExposedAceImmediately ? 1 : 0,
            buriedRisk, shallowerRisk);
    return knowledgeIsNotPhysical && revealAckMatches && opaqueDrawAckMatches &&
           drawChangesPhysical &&
           recoveryKeepsDistantKingPath &&
           canonicalColumns && compactKeyKeepsKnowledge &&
           compactKeyKeepsStockOrder && compactKeyKeepsWideIds &&
           drawPreservesNativeOrder && opensNovelTalonExit &&
           rejectsVisitedTalonExit && rejectsTalonWithoutExit &&
            opensLaterTalonExit && livenessFindsNovelExit &&
            certifiesNovelTalonBridge &&
           rejectsRepeatedTalonBridge &&
           tableauMoveKeepsReveal &&
           scoresUnmappedReveal && revealBeforeWaste &&
            rejectsCycle && acceptsReveal && rejectsDebtOnly &&
            rejectsFoundationRebuild && allowsFoundationAboveHighWater &&
            rejectsVisitedRecoveryPreparation &&
           ranksUncertainty && findsExactFinish && resumesFinishProof &&
           rejectsAfterWin &&
            rejectsBlockedFinish && recognizesNativeAutoLayout &&
            rejectsIncompleteAutoLayout && recycleAccounting &&
             catchesAceOnFirstOrbit && playsExposedAceImmediately &&
             kingRoutesUsefulExit;
}

static bool activateStrategicRecovery(const GameState& physicalState,
                                      const std::string& physicalKey) {
    DecisionCheckpoint* checkpoint = latestRecoveryCheckpoint();
    // If logical checkpoints are exhausted, progressively widen a bounded
    // perfect-information search from the CURRENT physical state. Never replay
    // an old board physically and never rebuild the same shallow frontier.
    if (!checkpoint && g_progressTracker.recoveryAttemptsAtState > 0) {
        int& escalation = g_recoveryEscalationByState[physicalKey];
        if (escalation < SOL_MAX_RECOVERY_ESCALATION) {
            ++escalation;
            ++g_progressTracker.recoveryAttemptsAtState;
            g_recoveryMode = true;
            g_recoveryCheckpointId = 0;
            g_lastRecoveryAt = nowMs();
            g_recoveryGraceTransitions = 0;
            g_committedPlan.clear();
            g_emergencyNoMoveKey.clear();
            g_emergencyEvaluationRootKey.clear();
            g_emergencyEvaluatedEdges.clear();
            g_exhaustedRecoveryKey.clear();
            g_exhaustedRecoveryAt = 0;
            g_progressTracker.stallPending = false;
            g_progressTracker.stallReason.clear();
            initializePersistentRoot(physicalState, physicalKey);
            SOL_LOG("[RECOVERY] checkpoint=0 escalation=%d/%d known=%d/52 depth=%d",
                    escalation, SOL_MAX_RECOVERY_ESCALATION,
                    knownCardCount(physicalState), g_persistentPlanner.maxDepth);
            return true;
        }
        g_recoveryMode = false;
        g_recoveryCheckpointId = 0;
        g_recoveryGraceTransitions = 0;
        g_committedPlan.clear();
        // Maximum recovery gets one fresh emergency pass, including the
        // tightly bounded Foundation bridge below.
        g_emergencyNoMoveKey.clear();
        g_emergencyEvaluationRootKey.clear();
        g_emergencyEvaluatedEdges.clear();
        g_exhaustedRecoveryKey = physicalKey;
        g_exhaustedRecoveryAt = nowMs();
        clearPlannerFrontier();
        g_progressTracker.stallPending = false;
        g_progressTracker.stallReason.clear();
        // Some rounds publish only a partial PlayerCardsMapping during the
        // opening sync window. Re-arm a bounded sync burst only after every
        // strategic branch at this exact state has failed. This lets the four
        // opaque values seen in the captured failure become available later
        // without polling forever during normal play.
        if (!g_mappingValidated && knownCardCount(physicalState) < 52 &&
            g_syncPlayerCardsMapping &&
            g_mappingSyncAttempts >= SOL_MAPPING_SYNC_MAX_ATTEMPTS &&
            g_mappingRecoveryBursts < SOL_MAPPING_SYNC_RECOVERY_BURSTS) {
            ++g_mappingRecoveryBursts;
            g_mappingSyncAttempts = 0;
            g_lastMappingSyncMs = 0;
            SOL_LOG("[MAPPING_RECOVERY] burst=%d/%d known=%d",
                    g_mappingRecoveryBursts,
                    SOL_MAPPING_SYNC_RECOVERY_BURSTS,
                    knownCardCount(physicalState));
        }
        SOL_LOG("[RECOVERY_RESULT] exhausted checkpoint=0 opaque=%d known=%d escalation=%d; cooldown=%llums",
                hasOpaqueCards(physicalState) ? 1 : 0,
                knownCardCount(physicalState), escalation,
                (unsigned long long)SOL_EXHAUSTED_RETRY_MS);
        return false;
    }

    uint64_t checkpointId = 0;
    std::string failedBranch = "none";
    int alternativesRemaining = 0;
    if (checkpoint) {
        checkpointId = checkpoint->id;
        failedBranch = checkpoint->selectedMoveKey;
        failStrategicCheckpoint(*checkpoint);
        alternativesRemaining = remainingCheckpointAlternatives(*checkpoint);
    }

    ++g_progressTracker.recoveryAttemptsAtState;
    g_recoveryMode = true;
    g_recoveryCheckpointId = checkpointId;
    g_lastRecoveryAt = nowMs();
    g_recoveryGraceTransitions = 0;
    g_committedPlan.clear();
    g_progressTracker.stallPending = false;
    SOL_LOG("[RECOVERY] checkpoint=%llu failedBranch=%s alternativesRemaining=%d",
            (unsigned long long)checkpointId, failedBranch.c_str(),
            alternativesRemaining);

    // Preserve a live current-root frontier and its sibling alternatives.
    // Reinitialise only when it belongs to another physical state or has been
    // fully exhausted; this is deliberately later than plan/alternative
    // recovery, never the first action.
    const bool currentPlannerUsable = g_persistentPlanner.rootKey == physicalKey &&
        (!g_persistentPlanner.frontier.empty() || !g_persistentPlanner.roots.empty() ||
         !g_persistentPlanner.recoveryRoots.empty());
    if (currentPlannerUsable) {
        g_persistentPlanner.maxDepth = std::min(320,
            std::max(g_persistentPlanner.maxDepth,
                     g_persistentPlanner.endgame ? 288 : 208));
    } else {
        initializePersistentRoot(physicalState, physicalKey);
    }
    return true;
}

static void detectPlannerIdleStall(const std::string& key, uint64_t now) {
    if (!g_progressTracker.initialized || g_progressTracker.stallPending ||
        g_waitingForMoveResult || !g_committedPlan.empty()) return;
    if (g_progressTracker.exactKey != key ||
        g_persistentPlanner.rootKey != key) return;
    const uint64_t idleStart = std::max(g_progressTracker.stateChangedAt,
                                        g_lastRecoveryAt);
    const uint64_t idleMs = now > idleStart ? now - idleStart : 0;
    const bool noPlan = g_persistentPlanner.roots.empty() &&
                        g_persistentPlanner.recoveryRoots.empty();
    if (!noPlan) return;
    if (g_persistentPlanner.frontier.empty() && idleMs >= 700) {
        flagStrategicStall("planner-frontier-exhausted");
    } else if (g_persistentPlanner.frames >= 8 &&
               idleMs >= (g_persistentPlanner.endgame ? 2600ULL : 1800ULL)) {
        flagStrategicStall("planner-no-executable-continuation");
    }
}

static bool physicalCandidateAllowed(const GameState& state, const Move& move, uint64_t now) {
    GameState after = state;
    if (!applyVirtualMove(after, move)) return false;

    if (move.kind == VirtualMoveKind::FoundationToTableau) {
        if (move.from < 0 || move.from >= 4 || state.foundations[move.from].empty())
            return false;
        const int cardId = state.foundations[move.from].back().id;
        if (cardId >= 0 && cardId < (int)g_foundationRescueEpoch.size() &&
            g_foundationRescueEpoch[cardId] == g_progressTracker.progressEpoch) {
            return false;
        }
    }

    // Keep the old exact-state guard for physical Tableau reversals. It is
    // intentionally not applied to a legal Foundation pullback because the
    // Foundation/Waste context may have changed and made that rescue useful.
    if (move.kind == VirtualMoveKind::Tableau && move.from >= 0 &&
        seenRecentSolverState(after, now)) {
        return false;
    }

    // The missing half of the old guard: columns are symmetric in Solitaire.
    // During one no-progress epoch, never execute a move that returns to a
    // canonical structure already reached under different column indices.
    // A reveal/Foundation advance starts a new epoch and remains admissible.
    const ProgressMetrics& epoch = g_progressTracker.baseline;
    const bool durable = durableProgressFrom(
        after,
        g_progressTracker.initialized ? epoch.hidden : hiddenCardCount(state),
        g_progressTracker.initialized ? epoch.foundation : virtualFoundationCount(state),
        g_progressTracker.initialized ? epoch.foundationDebt :
                                         foundationRiskDebt(state.foundations)).any();
    if (!durable &&
        canonicalStateVisitedSinceProgress(solverPositionKey(after))) {
        const bool priorityFoundationBridge =
            move.kind == VirtualMoveKind::Talon &&
            hasPriorityStockFoundationExit(state);
        std::vector<Move> kingRoute;
        const bool kingUsefulStockBridge =
            move.kind == VirtualMoveKind::Talon &&
            findKingStockExitLine(state, kingRoute) && !kingRoute.empty();
        const uint64_t unchangedMs = g_progressTracker.initialized &&
            now >= g_progressTracker.stateChangedAt
                ? now - g_progressTracker.stateChangedAt : 0;
        const bool boundedTalonBridge =
            move.kind == VirtualMoveKind::Talon &&
            g_talonBridgeAttempted.find(virtualPositionKey(state)) ==
                g_talonBridgeAttempted.end() &&
            (priorityFoundationBridge || kingUsefulStockBridge ||
             ((g_progressTracker.movesWithoutProgress >= 2 ||
               unchangedMs >= SOL_TALON_LIVENESS_MS) &&
              hasNovelTalonExit(state, after,
                                g_progressTracker.visitsSinceProgress,
                                unchangedMs < SOL_TALON_LIVENESS_MS)));
        if (!boundedTalonBridge) return false;
    }
    return true;
}

static bool executeExpertMove(void* localGs, const GameState& state,
                              const Move& move, int score) {
    g_lastDispatchOutcome = DispatchOutcome::Rejected;
    resetLastSubmissionObservation();
    const uint64_t now = nowMs();
    if (move.kind == VirtualMoveKind::Talon) {
        if (!g_talonUi || (state.talon.empty() && state.waste.empty())) return false;
        if (!submissionQueueHasRoom(localGs)) {
            g_lastDispatchOutcome = DispatchOutcome::Deferred;
            g_dispatchDeferredThisTick = true;
            g_dispatchRetryAtMs = now + SOL_PIPELINE_RETRY_MS;
            return false;
        }
        GameState afterDraw = state;
        if (applyVirtualMove(afterDraw, move) &&
            canonicalStateVisitedSinceProgress(solverPositionKey(afterDraw))) {
            // Record a permitted bridge before submitting the void UI call.
            // Even if Unity rejects it, the same root cannot spin forever.
            g_talonBridgeAttempted.insert(virtualPositionKey(state));
        }
        const int64_t sequenceBefore = localSequenceNumber(localGs);
        g_talonUi(nullptr);
        captureLastSubmissionObservation(localGs, sequenceBefore);
        g_lastDispatchOutcome = DispatchOutcome::Submitted;
        g_dispatchRetryPhysicalKey.clear();
        g_dispatchRetryAtMs = 0;
        g_lastActionWasTalon = true;
        if (state.talon.empty()) {
            ++g_actualRecycleCount;
            ++g_stagnantRecycleCount;
        }
        g_lastTalonMs = now;
        g_lastMoveMs = now;
        SOL_LOG("expert talon mode=%s score=%d stock=%d waste=%d recycles=%d charged=%d",
                state.talon.empty() ? "recycle" : "draw", score,
                (int)state.talon.size(), (int)state.waste.size(),
                g_actualRecycleCount,
                std::max(0, g_actualRecycleCount - SOL_FREE_TALON_RECYCLES));
        return true;
    }
    if (move.kind == VirtualMoveKind::Foundation) {
        if (move.to < 0 || move.to >= 4) return false;
        const Card* card = nullptr;
        std::string source;
        if (move.from < 0) {
            if (state.waste.empty()) return false;
            card = &state.waste.back();
            source = g_stackPrefix + "waste";
        } else {
            if (move.from >= 7 || state.tableau[move.from].empty()) return false;
            card = &state.tableau[move.from].back();
            source = tableauName(move.from);
        }
        if (!card->faceUp()) return false;
        const bool moved = moveCard(localGs, *card, source, foundationName(move.to),
                                    (int)state.foundations[move.to].size());
        if (moved) {
            g_stagnantRecycleCount = 0;
            SOL_LOG("expert foundation id=%d from=%d to=%d score=%d",
                    card->id, move.from, move.to, score);
        }
        return moved;
    }
    if (move.kind == VirtualMoveKind::FoundationToTableau) {
        if (move.from < 0 || move.from >= 4 || move.to < 0 || move.to >= 7 ||
            state.foundations[move.from].empty()) return false;
        const Card& card = state.foundations[move.from].back();
        const bool moved = moveCard(localGs, card, foundationName(move.from), tableauName(move.to),
                                    (int)state.tableau[move.to].size());
        if (moved) {
            // Commit the per-epoch pullback guard only after the expected board
            // is verified; a submitted/no-op native call must not spend it.
            g_lastSubmittedFoundationRescueCardId = card.id;
            SOL_LOG("expert foundation-rescue submitted id=%d from=%d to=%d score=%d",
                    card.id, move.from, move.to, score);
        }
        return moved;
    }
    if (move.to < 0 || move.to >= 7) return false;
    const Card* card = nullptr;
    std::string source;
    if (move.from < 0) {
        if (state.waste.empty()) return false;
        card = &state.waste.back();
        source = g_stackPrefix + "waste";
    } else {
        if (move.from >= 7 || move.start < 0 ||
            move.start >= (int)state.tableau[move.from].size()) return false;
        card = &state.tableau[move.from][move.start];
        source = tableauName(move.from);
    }
    if (!card->faceUp()) return false;
    const bool moved = moveCard(localGs, *card, source, tableauName(move.to),
                                (int)state.tableau[move.to].size());
    if (moved) {
        g_stagnantRecycleCount = 0;
        g_rrSource = move.from >= 0 ? (move.from + 1) % 8 : 0;
        SOL_LOG("expert tableau id=%d from=%d to=%d start=%d score=%d",
                card->id, move.from, move.to, move.start, score);
    }
    return moved;
}

// Emergency moves used to bypass the normal pending-action handshake.  A UI
// call could therefore report success while the physical board stayed exactly
// the same, after which the bot clicked the same Talon/card every 650 ms.  Route
// every escape move through the same expected-state validation as a normal
// planner move and remember the edge immediately, so a failed/no-op action can
// never become an infinite visual loop.
static bool executeTrackedEmergencyMove(void* localGs, const GameState& state,
                                        const Move& move, int score,
                                        const char* reason) {
    // A card view that is still settling is not a failed strategic edge.
    // Do not try a different emergency action from the same stale frame.
    if (g_dispatchDeferredThisTick) return false;
    const std::string rootKey = virtualPositionKey(state);
    const std::string edgeKey = moveKey(move);
    auto& tried = g_emergencyEdgesTried[rootKey];
    if (tried.find(edgeKey) != tried.end() || rootMoveFailed(rootKey, move))
        return false;
    if (!physicalCandidateAllowed(state, move, nowMs())) {
        tried.insert(edgeKey);
        return false;
    }

    GameState expected = state;
    if (!applyVirtualMove(expected, move)) {
        tried.insert(edgeKey);
        return false;
    }

    const int recycleCountBefore = g_actualRecycleCount;
    const int stagnantRecycleCountBefore = g_stagnantRecycleCount;
    if (!executeExpertMove(localGs, state, move, score)) {
        if (deferPhysicalDispatch(state)) return false;
        tried.insert(edgeKey);
        registerPhysicalRootFailure(rootKey, edgeKey);
        SOL_LOG("emergency submit-failed reason=%s move=%s",
                reason ? reason : "unknown", edgeKey.c_str());
        return false;
    }
    tried.insert(edgeKey);

    g_pendingRootKey = rootKey;
    g_pendingExpectedKey = virtualPositionKey(expected);
    g_pendingPhysicalRootKey = physicalPositionKey(state);
    g_pendingPhysicalExpectedKey = physicalPositionKey(expected);
    g_pendingMoveKey = edgeKey;
    g_pendingMoveAt = nowMs();
    g_pendingRecycleCountBefore = recycleCountBefore;
    g_pendingStagnantRecycleCountBefore = stagnantRecycleCountBefore;
    armPendingFlightObservation();
    g_waitingForMoveResult = true;
    scheduleAnticipatorySearch(expected);
    SOL_LOG("emergency selected reason=%s move=%s score=%d expected=%s",
            reason ? reason : "unknown", edgeKey.c_str(), score,
            g_pendingExpectedKey.c_str());
    return true;
}

// Search remains the authority for card moves.  Talon is different: it is a
// reversible reachability action, and waiting for a still-live best-first
// frontier can make the board look frozen even though the next stock exposure
// has a proven new exit.  After the bounded liveness window, submit exactly one
// such draw through the normal physical ACK path.  Published progress plans
// always win; this fallback runs only while both plan sets are empty.
static bool attemptProvenTalonLiveness(void* localGs, const GameState& state,
                                       uint64_t now) {
    if (g_waitingForMoveResult || !g_committedPlan.empty() ||
        !g_persistentPlanner.roots.empty() ||
        !g_persistentPlanner.recoveryRoots.empty() ||
        !g_progressTracker.initialized ||
        g_progressTracker.exactKey != virtualPositionKey(state) ||
        now < g_progressTracker.stateChangedAt ||
        now - g_progressTracker.stateChangedAt < SOL_TALON_LIVENESS_MS ||
        (state.talon.empty() && state.waste.empty())) {
        return false;
    }

    const Move draw{VirtualMoveKind::Talon, -1, -1, -1};
    GameState after = state;
    if (!applyVirtualMove(after, draw)) return false;
    const bool priorityFoundation = hasPriorityStockFoundationExit(state);
    const bool novelReachableExit = hasNovelTalonExit(
        state, after, g_progressTracker.visitsSinceProgress, false);
    if (!priorityFoundation && !novelReachableExit) return false;
    if (!physicalCandidateAllowed(state, draw, now)) return false;

    const char* proof = priorityFoundation
        ? "talon-liveness-foundation" : "talon-liveness-novel-exit";
    if (!executeTrackedEmergencyMove(localGs, state, draw, -120000, proof))
        return false;
    SOL_LOG("[TALON_LIVENESS] proof=%s idle=%llu frontier=%d expanded=%d",
            proof,
            (unsigned long long)(now - g_progressTracker.stateChangedAt),
            (int)g_persistentPlanner.frontier.size(),
            g_persistentPlanner.expanded);
    return true;
}

static bool attemptEmergencyProgress(void* localGs, const GameState& state) {
    // This path is entered only after the retained planner has no executable
    // root (or after several no-progress stock passes).  It deliberately
    // performs only locally proven legal moves; its purpose is to prevent a
    // live board from becoming visually frozen because a search branch was
    // exhausted or over-penalised.

    const std::string rootKey = virtualPositionKey(state);
    if (g_emergencyNoMoveKey == rootKey) return false;
    if (g_emergencyEvaluationRootKey != rootKey) {
        g_emergencyEvaluationRootKey = rootKey;
        g_emergencyEvaluatedEdges.clear();
    }

    // First free a hidden Tableau card by advancing its exposed blocker to a
    // matching foundation.  This is stronger than another blind stock pass,
    // even when the normal conservative foundation threshold would defer it.
    for (int src = 0; src < 7; ++src) {
        if (state.tableau[src].empty()) continue;
        const Card& card = state.tableau[src].back();
        if (!card.faceUp()) continue;
        const bool revealsHidden = state.tableau[src].size() > 1 &&
                                   !state.tableau[src][state.tableau[src].size() - 2].faceUp();
        if (!revealsHidden && !foundationAdvanceIsSafe(card, state.foundations)) continue;
        for (int dst = 0; dst < 4; ++dst) {
            const Card* top = state.foundations[dst].empty() ? nullptr : &state.foundations[dst].back();
            if (!canDropOnFoundation(card, top)) continue;
            const Move move{VirtualMoveKind::Foundation, src, dst, -1};
            if (executeTrackedEmergencyMove(localGs, state, move,
                                            revealsHidden ? 180000 : 120000,
                                            revealsHidden ? "foundation-reveal" :
                                                            "safe-foundation")) {
                return true;
            }
        }
    }

    // Only Draw-1 may treat this as unconditional Foundation progress. In
    // Draw-3, keep the legal Waste move in the searched candidates below so
    // its altered stock alignment is compared with the other continuations.
    if (state.drawCount == 1 && !state.waste.empty() && state.waste.back().faceUp()) {
        const Card& card = state.waste.back();
        // An emergency must not create new Foundation debt merely because it
        // is the first legal move. Unsafe advances remain available only to
        // searched recovery when their continuation proves useful.
        if (foundationAdvanceIsSafe(card, state.foundations)) {
            for (int dst = 0; dst < 4; ++dst) {
                const Card* top = state.foundations[dst].empty() ? nullptr : &state.foundations[dst].back();
                if (!canDropOnFoundation(card, top)) continue;
                const Move move{VirtualMoveKind::Foundation, -1, dst, -1};
                if (executeTrackedEmergencyMove(localGs, state, move, 100000,
                                                "waste-foundation")) {
                    return true;
                }
            }
        }
    }

    // A Waste -> Tableau move is still available to the searched-preparation
    // candidates below. Never dispatch it greedily before a direct hidden
    // reveal or before comparing it with other legal preparation branches.
    Move revealMove;
    if (findEmergencyTableauReveal(state, revealMove) &&
        executeTrackedEmergencyMove(localGs, state, revealMove, 160000,
                                    "tableau-reveal")) {
        return true;
    }

    // The strategic planner intentionally records only lines that prove a
    // reveal, safe Foundation gain, debt reduction, or a strong recovery
    // preparation.  A rare live position can still have legal preparatory
    // moves after every such branch is exhausted.  Rank those moves with the
    // existing virtual look-ahead instead of parking the bot or choosing one
    // greedily.  This block is reached only after normal planning and all
    // checkpoint recovery have failed, so it cannot disturb the strong common
    // path that already wins most games.
    struct EmergencyCandidate {
        Move move;
        int preliminary = -10000000;
        int future = -10000000;
        std::string edgeKey;
        std::string resultKey;
        bool foundationPullback = false;
    };
    std::vector<Move> legal;
    const bool perfectRecovery = knownCardCount(state) == 52 && !hasOpaqueCards(state);
    // At maximum recovery, a Foundation pullback may be the only legal bridge
    // that recreates a Tableau build target and exposes one of the opaque
    // cards. Admit it only after sustained no-progress; the per-card epoch
    // guard prevents a Foundation/Tableau ping-pong loop.
    const auto escalationIt = g_recoveryEscalationByState.find(rootKey);
    const int recoveryEscalation = escalationIt == g_recoveryEscalationByState.end()
        ? 0 : escalationIt->second;
    const bool allowEmergencyPullback = perfectRecovery ||
        (recoveryEscalation >= SOL_MAX_RECOVERY_ESCALATION &&
         g_progressTracker.movesWithoutProgress >= 8);
    const bool preparationBudgetOpen =
        g_progressTracker.emergencyPreparations <
        SOL_MAX_EMERGENCY_PREPARATIONS;
    generateVirtualMoves(state, legal, allowEmergencyPullback,
                         g_recoveryMode && perfectRecovery);
    std::vector<EmergencyCandidate> candidates;
    candidates.reserve(legal.size());
    for (const Move& move : legal) {
        if (move.kind == VirtualMoveKind::Talon) continue;
        const bool foundationPullback =
            move.kind == VirtualMoveKind::FoundationToTableau;
        // Once the bounded preparation budget is spent, do not keep walking
        // through fresh-looking permutations of the same blocked Tableau.
        // Preserve only the deliberately guarded Foundation bridge: it is the
        // one action that can recreate a missing build target, and each card
        // can already be pulled at most once per real-progress epoch.
        if (!foundationPullback && !preparationBudgetOpen) continue;
        const std::string edgeKey = moveKey(move);
        if (g_emergencyEvaluatedEdges.find(edgeKey) !=
            g_emergencyEvaluatedEdges.end()) {
            continue;
        }
        const auto triedAtRoot = g_emergencyEdgesTried.find(rootKey);
        if ((triedAtRoot != g_emergencyEdgesTried.end() &&
             triedAtRoot->second.find(edgeKey) != triedAtRoot->second.end()) ||
            rootMoveFailed(rootKey, move) ||
            !physicalCandidateAllowed(state, move, nowMs())) {
            continue;
        }
        GameState after = state;
        if (!applyVirtualMove(after, move)) continue;
        const std::string resultKey = virtualPositionKey(after);
        // Exact repetition is a proven motion loop, not a useful rescue.
        if (seenRecentSolverState(after, nowMs())) continue;
        if (g_deadStateCache.find(solverPositionKey(after)) != g_deadStateCache.end())
            continue;
        const int preliminary = virtualStructuralScore(after) +
                                candidateTacticalBonus(state, after, move) +
                                immediateMoveTier(state, move) * 110 +
                                (foundationPullback && !preparationBudgetOpen
                                     ? 180000 : 0) -
                                strategicStatePenalty(solverPositionKey(after));
        candidates.push_back({move, preliminary, -10000000, edgeKey, resultKey,
                              foundationPullback});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const EmergencyCandidate& a, const EmergencyCandidate& b) {
                  return a.preliminary > b.preliminary;
              });

    // Share one small deadline among the best candidates.  Rescue search is
    // deliberately bounded because this runs on the Unity game thread.
    const uint64_t rescueDeadline = nowMs() + 20;
    const int evaluateCount = std::min(8, (int)candidates.size());
    int evaluatedCount = 0;
    for (; evaluatedCount < evaluateCount && nowMs() < rescueDeadline;
         ++evaluatedCount) {
        const int depth = hiddenCardCount(state) <= 7 ? 20 : 16;
        candidates[evaluatedCount].future = scoreFutureCandidate(
            state, candidates[evaluatedCount].move, depth, 480, rescueDeadline,
            perfectRecovery, g_recoveryMode && perfectRecovery);
        if (candidates[evaluatedCount].future > -8000000) {
            candidates[evaluatedCount].future +=
                immediateMoveTier(state, candidates[evaluatedCount].move) * 140;
        } else {
            g_emergencyEvaluatedEdges.insert(candidates[evaluatedCount].edgeKey);
        }
    }
    std::sort(candidates.begin(), candidates.begin() + evaluatedCount,
              [](const EmergencyCandidate& a, const EmergencyCandidate& b) {
                  if (a.future != b.future) return a.future > b.future;
                  return a.preliminary > b.preliminary;
              });
    for (int i = 0; i < evaluatedCount; ++i) {
        if (candidates[i].future <= -8000000) continue;
        if (executeTrackedEmergencyMove(localGs, state, candidates[i].move,
                                         candidates[i].future,
                                         "searched-preparation")) {
            if (!candidates[i].foundationPullback)
    g_pendingCountsEmergencyPreparation = true;
            SOL_LOG("emergency searched-choice rank=%d/%d preliminary=%d result=%s",
                    i + 1, evaluatedCount, candidates[i].preliminary,
                    candidates[i].resultKey.c_str());
            return true;
        }
        // A temporarily missing controller is not a rejected strategic edge.
        if (g_dispatchDeferredThisTick) return false;
        g_emergencyEvaluatedEdges.insert(candidates[i].edgeKey);
    }
    const bool hasUnevaluatedCandidates =
        evaluatedCount < (int)candidates.size();
    if (hasUnevaluatedCandidates) {
        // Do not let the Talon fallback change the root before later batches
        // get their turn; that was the original first-eight-only blind spot.
        SOL_LOG("emergency batch partial evaluated=%d remaining=%d root=%s",
                evaluatedCount, (int)candidates.size() - evaluatedCount,
                rootKey.c_str());
        return false;
    }

    // A completely explored planning frontier must never freeze with stock
    // cards still physically drawable. Advance to the next *new* Talon state;
    // exact-state history blocks a useless full recycle, so this escape hatch
    // cannot become the old infinite stock loop.
    if (!state.talon.empty() || !state.waste.empty()) {
        const Move talonMove{VirtualMoveKind::Talon, -1, -1, -1};
        GameState next = state;
        if (applyVirtualMove(next, talonMove) &&
            physicalCandidateAllowed(state, talonMove, nowMs()) &&
            executeTrackedEmergencyMove(localGs, state, talonMove, -500000,
                                        "talon-new-state")) {
            SOL_LOG("emergency talon advance stock=%d waste=%d",
                    (int)state.talon.size(), (int)state.waste.size());
            return true;
        }
    }
    // Memoize no-action only after every generated preparation was actually
    // evaluated. A batch/deadline boundary now resumes on the next frame.
    if (!g_dispatchDeferredThisTick) {
        g_emergencyNoMoveKey = rootKey;
    }
    return false;
}

static void logStalledExpertState(const GameState& state, uint64_t now) {
    if (now - g_lastStallDiagMs < 2500) return;
    g_lastStallDiagMs = now;
    std::string tableauTops;
    std::string foundationTops;
    for (int i = 0; i < 7; ++i) {
        if (!tableauTops.empty()) tableauTops += ',';
        if (state.tableau[i].empty()) tableauTops += "-";
        else tableauTops += std::to_string(state.tableau[i].back().id) + ":" +
                            std::to_string(state.tableau[i].back().value);
    }
    for (int i = 0; i < 4; ++i) {
        if (!foundationTops.empty()) foundationTops += ',';
        if (state.foundations[i].empty()) foundationTops += "-";
        else foundationTops += std::to_string(state.foundations[i].back().id) + ":" +
                               std::to_string(state.foundations[i].back().value);
    }
    std::vector<Move> legal;
    generateVirtualMoves(state, legal);
    const int wasteId = state.waste.empty() ? -1 : state.waste.back().id;
    const int wasteValue = state.waste.empty() ? -1 : state.waste.back().value;
#if GOAT_SOLITAIRE_TRACE
    int allowed = 0;
    int allowedTalon = 0;
    for (const Move& move : legal) {
        if (!physicalCandidateAllowed(state, move, now)) continue;
        ++allowed;
        if (move.kind == VirtualMoveKind::Talon) ++allowedTalon;
    }
    const std::string exactKey = virtualPositionKey(state);
    const auto tried = g_emergencyEdgesTried.find(exactKey);
    const int triedCount = tried == g_emergencyEdgesTried.end()
        ? 0 : (int)tried->second.size();
    SOL_LOG("[DIAG] known=%d/52 legal=%d allowed=%d talonAllowed=%d prep=%d/%d noMoveLatch=%d tried=%d visits=%d expanded=%d transpositions=%d progressEpoch=%d withoutProgress=%d pending=%d",
            knownCardCount(state), (int)legal.size(), allowed, allowedTalon,
            g_progressTracker.emergencyPreparations,
            SOL_MAX_EMERGENCY_PREPARATIONS,
            g_emergencyNoMoveKey == exactKey ? 1 : 0,
            triedCount, (int)g_progressTracker.visitsSinceProgress.size(),
            g_persistentPlanner.expanded,
            (int)g_persistentPlanner.bestPathCost.size(),
            g_progressTracker.progressEpoch,
            g_progressTracker.movesWithoutProgress,
            g_waitingForMoveResult ? 1 : 0);
    // One exact board dump per distinct stalled position is enough to replay
    // candidate legality offline. Never serialize every frame or search node.
    static std::string lastStalledBoard;
    if (lastStalledBoard != exactKey) {
        lastStalledBoard = exactKey;
        const auto logPile = [](const char* name, const std::vector<Card>& pile) {
            std::string cards;
            cards.reserve(pile.size() * 10);
            for (const Card& card : pile) {
                if (!cards.empty()) cards.push_back(',');
                cards += std::to_string(card.id) + ':' +
                    std::to_string(card.known() ?
                        (card.faceUp() ? card.value : card.knownValue) : -1);
                cards.push_back(card.faceUp() ? 'u' : 'd');
            }
            SOL_LOG("[DIAG] pile=%s cards=[%s]", name, cards.c_str());
        };
        for (int i = 0; i < 7; ++i) {
            const std::string name = "tableau_" + std::to_string(i);
            logPile(name.c_str(), state.tableau[i]);
        }
        logPile("talon", state.talon);
        logPile("waste", state.waste);
        for (int i = 0; i < 4; ++i) {
            const std::string name = "foundation_" + std::to_string(i);
            logPile(name.c_str(), state.foundations[i]);
        }
    }
#endif
    SOL_LOG("expert stalled recycle=%d draw=%d stock=%d waste=%d top=%d:%d moves=%d frontier=%d roots=%d risky=%d recoveryRoots=%d recoveryMode=%d committed=%d debt=%d tableau=[%s] foundation=[%s]",
            state.recycles, state.drawCount, (int)state.talon.size(), (int)state.waste.size(),
            wasteId, wasteValue, (int)legal.size(),
            (int)g_persistentPlanner.frontier.size(),
            (int)g_persistentPlanner.roots.size(),
            0,
            (int)g_persistentPlanner.recoveryRoots.size(), g_recoveryMode ? 1 : 0,
            (int)g_committedPlan.size(), foundationRiskDebt(state.foundations),
            tableauTops.c_str(), foundationTops.c_str());
}

static int immediateMoveTier(const GameState& state, const Move& move) {
    if (move.kind == VirtualMoveKind::FoundationToTableau) return -100;
    if (move.kind == VirtualMoveKind::Talon) return 0;
    if (move.kind == VirtualMoveKind::Foundation) {
        const Card* card = nullptr;
        if (move.from < 0) {
            if (!state.waste.empty()) card = &state.waste.back();
        } else if (move.from < 7 && !state.tableau[move.from].empty()) {
            card = &state.tableau[move.from].back();
        }
        if (!card) return -100;
        if (move.from >= 0 && state.tableau[move.from].size() > 1 &&
            !state.tableau[move.from][state.tableau[move.from].size() - 2].faceUp())
            return 110; // foundation advance that immediately uncovers a card
        // Aces/twos are unconditional progress; other safe foundation moves
        // remain above stock cycling but below revealing a hidden card.
        if (cardRank(*card) <= 1) return 90;
        if (foundationAdvanceIsSafe(*card, state.foundations)) return 70;
        // A legal exposed waste card may need to move upward to uncover the
        // next draw-3 card. Keep it available as a rescue, not forbidden.
        return move.from < 0 ? 50 : 0;
    }
    if (move.from < 0) return 80; // playable waste card must not be skipped.
    if (move.from >= 7 || move.to < 0 || move.to >= 7 || move.start < 0 ||
        move.start >= (int)state.tableau[move.from].size()) return -100;
    const bool revealsHidden = move.start > 0 && !state.tableau[move.from][move.start - 1].faceUp();
    if (revealsHidden) return 100;
    if (improvesNextPosition(state.tableau, state.foundations, move.from, move.to,
                             state.tableau[move.from][move.start].id)) return 60;
    return 0;
}

// Exact local action-score rules recovered from SolitaireGameScore in the
// shipped IL2CPP build. They are deliberately a small tie-break only: survival
// proof, hidden-card progress and Foundation balance remain authoritative.
static int solitaireCardActionScore(const Card& card) {
    static constexpr std::array<int, 13> scores{
        90, 80, 70, 60, 50, 40, 30, 20, 10, 10, 10, 10, 120
    };
    const int value = card.faceUp() ? card.value : card.knownValue;
    return value >= 0 && value < 52 ? scores[(size_t)(value % 13)] : 0;
}

static int exactLocalScoreDelta(const GameState& before, const Move& move) {
    int delta = virtualRevealGain(before, move) * 20;
    if (move.kind == VirtualMoveKind::Foundation) {
        const Card* card = nullptr;
        if (move.from < 0) {
            if (!before.waste.empty()) card = &before.waste.back();
        } else if (move.from < 7 && !before.tableau[move.from].empty()) {
            card = &before.tableau[move.from].back();
        }
        if (card) delta += solitaireCardActionScore(*card);
    } else if (move.kind == VirtualMoveKind::FoundationToTableau) {
        if (move.from >= 0 && move.from < 4 &&
            !before.foundations[move.from].empty()) {
            delta -= solitaireCardActionScore(before.foundations[move.from].back());
        }
    } else if (move.kind == VirtualMoveKind::Tableau && move.from < 0 &&
               !before.waste.empty() && cardRank(before.waste.back()) != 12) {
        delta += 20;
    }
    return delta;
}

struct PlanTempoValue {
    int localScoreDelta = 0;
    int actions = 0;
    int talonActions = 0;
    int paidRecycles = 0;
};

static PlanTempoValue valueCertifiedPrefix(const GameState& root,
                                           const std::vector<Move>& line) {
    PlanTempoValue value;
    GameState cursor = root;
    const int rootHidden = hiddenCardCount(root);
    const int rootFoundation = virtualFoundationCount(root);
    const int rootDebt = foundationRiskDebt(root.foundations);
    for (const Move& move : line) {
        value.localScoreDelta += exactLocalScoreDelta(cursor, move);
        const int paidBefore = std::max(0, cursor.recycles -
                                        SOL_FREE_TALON_RECYCLES);
        if (move.kind == VirtualMoveKind::Talon) ++value.talonActions;
        if (!applyVirtualMove(cursor, move)) break;
        const int paidAfter = std::max(0, cursor.recycles -
                                       SOL_FREE_TALON_RECYCLES);
        const int newlyPaid = std::max(0, paidAfter - paidBefore);
        value.paidRecycles += newlyPaid;
        value.localScoreDelta -= newlyPaid * SOL_CHARGED_RECYCLE_SCORE_DELTA;
        ++value.actions;
        if (durableProgressFrom(cursor, rootHidden, rootFoundation, rootDebt).any())
            break;
    }
    return value;
}

static bool runApexScoreSelfTests() {
    Card ace;
    ace.id = 700;
    ace.value = ace.knownValue = 12;
    GameState aceFromWaste;
    aceFromWaste.waste.push_back(ace);
    const bool aceScores120 = exactLocalScoreDelta(
        aceFromWaste, {VirtualMoveKind::Foundation, -1, 0, -1}) == 120;

    Card redSix;
    redSix.id = 701;
    redSix.value = redSix.knownValue = 18;
    Card blackFive;
    blackFive.id = 702;
    blackFive.value = blackFive.knownValue = 4;
    GameState wasteToTableau;
    wasteToTableau.tableau[0].push_back(redSix);
    wasteToTableau.waste.push_back(blackFive);
    const bool wastePlacementScores20 = exactLocalScoreDelta(
        wasteToTableau, {VirtualMoveKind::Tableau, -1, 0, -1}) == 20;

    GameState pullback;
    pullback.foundations[0].push_back(ace);
    Card redTwo;
    redTwo.id = 703;
    redTwo.value = redTwo.knownValue = 13;
    pullback.tableau[0].push_back(redTwo);
    const bool pullbackDebits120 = exactLocalScoreDelta(
        pullback, {VirtualMoveKind::FoundationToTableau, 0, 0, -1}) == -120;

    GameState reveal;
    Card hidden;
    hidden.id = 704;
    hidden.knownValue = 8;
    reveal.tableau[0] = {hidden, blackFive};
    reveal.tableau[1] = {redSix};
    const Move revealMove{VirtualMoveKind::Tableau, 0, 1, 1};
    const bool revealScores20 = exactLocalScoreDelta(reveal, revealMove) == 20;
    const PlanTempoValue tempo = valueCertifiedPrefix(reveal, {revealMove});
    const bool prefixStopsAtProgress = tempo.actions == 1 &&
        tempo.talonActions == 0 && tempo.localScoreDelta == 20;

    SOL_LOG("[APEX_SCORE_TEST] foundation=%d waste=%d pullback=%d reveal=%d prefix=%d",
            aceScores120 ? 1 : 0, wastePlacementScores20 ? 1 : 0,
            pullbackDebits120 ? 1 : 0, revealScores20 ? 1 : 0,
            prefixStopsAtProgress ? 1 : 0);
    return aceScores120 && wastePlacementScores20 && pullbackDebits120 &&
           revealScores20 && prefixStopsAtProgress;
}

static void clearPendingSubmissionMetadata() {
    g_pendingFoundationRescueCardId = -1;
    g_pendingCountsEmergencyPreparation = false;
    g_pendingCountsRecoveryPreparation = false;
    g_pendingCheckpointId = 0;
}

static void discardPendingSubmissionEffects() {
    discardUnverifiedCheckpoint(g_pendingCheckpointId);
    clearPendingSubmissionMetadata();
}

static void commitPendingSubmissionEffects() {
    if (g_pendingFoundationRescueCardId >= 0 &&
        g_pendingFoundationRescueCardId <
            (int)g_foundationRescueEpoch.size()) {
        g_foundationRescueEpoch[g_pendingFoundationRescueCardId] =
            g_progressTracker.progressEpoch;
    }
    if (g_pendingCountsEmergencyPreparation)
        ++g_progressTracker.emergencyPreparations;
    if (g_pendingCountsRecoveryPreparation) {
        ++g_progressTracker.recoveryPreparations;
        SOL_LOG("[RECOVERY] preparation=%d/%d epoch=%d",
                g_progressTracker.recoveryPreparations,
                SOL_MAX_RECOVERY_PREPARATIONS,
                g_progressTracker.progressEpoch);
    }
    clearPendingSubmissionMetadata();
}

// ---- Server-mode pipelining ------------------------------------------------

// Call before discardPendingSubmissionEffects(): checkpoints are a stack and
// discardUnverifiedCheckpoint() only removes the newest one.
static void discardChainedSteps() {
    if (g_chainedSteps.empty()) return;
    for (auto it = g_chainedSteps.rbegin(); it != g_chainedSteps.rend(); ++it)
        discardUnverifiedCheckpoint(it->checkpointId);
    SOL_LOG("[PIPELINE] discarded queued=%d; draining %llums",
            (int)g_chainedSteps.size(), (unsigned long long)SOL_PIPELINE_DRAIN_MS);
    g_chainedSteps.clear();
    g_pipelineDrainUntilMs = nowMs() + SOL_PIPELINE_DRAIN_MS;
}

// Newest chained step whose result the physical board already shows, or -1.
static int chainedStepMatching(const std::string& physicalKey) {
    for (int i = (int)g_chainedSteps.size() - 1; i >= 0; --i) {
        if (g_chainedSteps[(size_t)i].physicalExpectedKey == physicalKey) return i;
    }
    return -1;
}

// True when every card the board would show is known, so the next plan step
// can be computed without seeing the server's answer. A step that uncovers
// an unresolved Tableau card (or draws unknown Waste cards) ends the queue.
static bool predictionIsExact(const GameState& state) {
    for (const auto& column : state.tableau) {
        for (const Card& card : column)
            if (card.virtuallyRevealed && !card.known()) return false;
    }
    for (const Card& card : state.waste) if (!card.known()) return false;
    return true;
}

// The pending flight has been verified, and so have chained steps 0..hit
// (hit = -1: none of them yet). Promote the next unverified step to be the
// pending flight. Always returns false so the tick admits the observed board
// to strategic history, exactly as after a single verified action.
static bool settleChainedSteps(int hit, const std::string& rootKey) {
    std::string verifiedKey = g_pendingExpectedKey;
    std::string verifiedPhysicalKey = g_pendingPhysicalExpectedKey;
    for (int i = 0; i <= hit && !g_chainedSteps.empty(); ++i) {
        verifiedKey = g_chainedSteps.front().expectedKey;
        verifiedPhysicalKey = g_chainedSteps.front().physicalExpectedKey;
        g_chainedSteps.pop_front();
    }
    if (rootKey != verifiedKey) {
        // Same rule as a single flight: learned values changed the position,
        // so the unsent remainder is replanned from the real board.
        g_committedPlan.clear();
        g_committedPlanAllowsRecoveryPreparation = false;
        g_committedPlanIsVerifiedWin = false;
    }
    if (g_chainedSteps.empty()) return false;

    ChainedStep next = std::move(g_chainedSteps.front());
    g_chainedSteps.pop_front();
    g_pendingRootKey = verifiedKey;
    g_pendingExpectedKey = next.expectedKey;
    g_pendingPhysicalRootKey = verifiedPhysicalKey;
    g_pendingPhysicalExpectedKey = next.physicalExpectedKey;
    g_pendingMoveKey = next.moveKey;
    g_pendingMoveAt = next.sentAt;
    g_pendingRecycleCountBefore = next.recycleCountBefore;
    g_pendingStagnantRecycleCountBefore = next.stagnantRecycleCountBefore;
    armPendingFlightObservation();
    g_pendingFoundationRescueCardId = -1;
    g_pendingCheckpointId = next.checkpointId;
    g_pendingExpectedState = std::move(next.expected);
    g_pendingChainable = true;
    g_pendingFlipsTableau = next.flipsTableau;
    g_waitingForMoveResult = true;
    SOL_LOG("[PIPELINE] promoted move=%s queued=%d",
            g_pendingMoveKey.c_str(), (int)g_chainedSteps.size());
    return false;
}

// Send the next committed-plan step behind the actions already in flight.
// Applies the same legality, proof and dead-state gates as the ordinary
// committed-plan dispatch, but to the predicted board after those actions.
static bool extendServerPipeline(void* localGs, uint64_t now,
                                 const std::string& observedPhysicalKey) {
    if (!g_waitingForMoveResult || !g_pendingChainable || humanSpeedMode()) return false;
    // A local sequence/PendingMove means the game applies moves itself
    // (optimistic mode); that path keeps its own queue and gating.
    if (g_pendingWasQueued || g_pendingSequence >= 0) return false;
    if ((int)g_chainedSteps.size() + 1 >= SOL_SERVER_PIPELINE_MAX) return false;
    if (g_committedPlan.empty() || g_dispatchDeferredThisTick) return false;
    // Queue more only while the real board is still on the predicted path;
    // an off-path board is about to be reconciled as an interruption.
    if (observedPhysicalKey != g_pendingPhysicalRootKey &&
        observedPhysicalKey != g_pendingPhysicalExpectedKey &&
        chainedStepMatching(observedPhysicalKey) < 0) return false;

    const bool tailFlipsTableau = g_chainedSteps.empty()
        ? g_pendingFlipsTableau : g_chainedSteps.back().flipsTableau;
    if (tailFlipsTableau) return false;
    const GameState& base = g_chainedSteps.empty()
        ? g_pendingExpectedState : g_chainedSteps.back().expected;
    if (!predictionIsExact(base)) return false;
    const Move next = g_committedPlan.front();
    // A Foundation pullback commits a per-epoch guard on ACK; keep it single.
    if (next.kind == VirtualMoveKind::FoundationToTableau) return false;

    GameState expected = base;
    if (!applyVirtualMove(expected, next)) return false;
    // Every queued step must leave a distinct layout, or the ACK could not
    // tell how far the server has got (e.g. a tiny stock cycling back).
    const std::string physicalExpected = physicalPositionKey(expected);
    if (physicalExpected == g_pendingPhysicalRootKey ||
        physicalExpected == g_pendingPhysicalExpectedKey) return false;
    for (const ChainedStep& step : g_chainedSteps) {
        if (step.physicalExpectedKey == physicalExpected) return false;
    }

    const bool certifiedWin = committedWinMatches(base);
    const bool proofSafe = certifiedWin || !completeKnownDeck(expected) ||
        !provenDependencyDeadEnd(expected);
    if (!proofSafe) return false;
    if (!certifiedWin &&
        (!physicalCandidateAllowed(base, next, now) ||
         g_deadStateCache.find(solverPositionKey(expected)) != g_deadStateCache.end()))
        return false;

    const int recycleCountBefore = g_actualRecycleCount;
    const int stagnantRecycleCountBefore = g_stagnantRecycleCount;
    if (!executeExpertMove(localGs, base, next, g_committedPlanScore)) {
        // A card view still settling is retried next tick; anything else
        // means this plan step cannot be sent ahead of the ACK.
        if (g_lastDispatchOutcome != DispatchOutcome::Deferred) g_pendingChainable = false;
        return false;
    }
    const uint64_t checkpointId = saveDecisionCheckpoint(
        base, next, g_committedPlan, legalAlternativeKeys(base, next),
        g_committedPlanScore);
    g_committedPlan.erase(g_committedPlan.begin());
    if (g_committedPlan.empty()) g_committedPlanIsVerifiedWin = false;
    trackCommittedWinState(expected);

    ChainedStep step;
    step.expectedKey = virtualPositionKey(expected);
    step.physicalExpectedKey = physicalExpected;
    step.moveKey = moveKey(next);
    step.sentAt = now;
    step.recycleCountBefore = recycleCountBefore;
    step.stagnantRecycleCountBefore = stagnantRecycleCountBefore;
    step.checkpointId = checkpointId;
    step.flipsTableau = hiddenCardCount(expected) < hiddenCardCount(base);
    step.expected = std::move(expected);
    g_chainedSteps.push_back(std::move(step));
    scheduleAnticipatorySearch(g_chainedSteps.back().expected);
    SOL_LOG("[PIPELINE] queued move=%s inFlight=%d remaining=%d",
            g_chainedSteps.back().moveKey.c_str(),
            (int)g_chainedSteps.size() + 1, (int)g_committedPlan.size());
    return true;
}

// LogMove can add its PendingMove after the native call returns. A queue miss
// at submission time therefore cannot prove that the action is unqueued.
// Bind only the immediately following sequence; a later sequence could belong
// to unrelated player input and must not be claimed as this bot action.
static void refreshPendingQueueObservation(void* localGs) {
    if (!localGs || g_pendingWasQueued) return;
    int64_t sequence = g_pendingSequence;
    if (sequence < 0 && g_pendingSequenceBefore >= 0) {
        const int64_t current = localSequenceNumber(localGs);
        if (current == g_pendingSequenceBefore + 1) sequence = current;
    }
    if (sequence >= 0 && pendingMovesContainSequence(localGs, sequence)) {
        g_pendingSequence = sequence;
        g_pendingWasQueued = true;
    }
}

// Reconcile one submitted native action before the observed board is admitted
// to strategic history. Transient animation/sync snapshots must not advance a
// progress epoch or poison loop detection while the flight is still unsettled.
static bool pendingFlightBlocksObservation(void* localGs,
                                           const GameState& state,
                                           uint64_t now) {
    if (!g_waitingForMoveResult) return false;
    const std::string rootKey = virtualPositionKey(state);
    const std::string physicalKey = physicalPositionKey(state);
    const uint64_t flightAge = now - g_pendingMoveAt;
    refreshPendingQueueObservation(localGs);
    const int pendingNow = listSize(localPendingMoves(localGs));
    const bool trackedQueueOutstanding = g_pendingWasQueued &&
        pendingMovesContainSequence(localGs, g_pendingSequence);

    if (physicalKey == g_pendingPhysicalRootKey) {
        g_pendingObservedPhysicalKey.clear();
        g_pendingObservedSinceMs = 0;
        if (flightAge < SOL_ACTION_RESULT_TIMEOUT_MS) return true;
        if (!g_pendingSoftTimeoutLogged) {
            g_pendingSoftTimeoutLogged = true;
            const bool exactSeqQueued =
                pendingMovesContainSequence(localGs, g_pendingSequence);
            SOL_LOG("[ACTION_UNCONFIRMED] move=%s age=%llu seq=%lld currentSeq=%lld pending=%d->%d queuedAtSubmit=%d queuedNow=%d localSync=%d ui=%d",
                    g_pendingMoveKey.c_str(),
                    (unsigned long long)flightAge,
                    (long long)g_pendingSequence,
                    (long long)localSequenceNumber(localGs),
                    g_pendingCountAtSubmit, pendingNow,
                    g_pendingWasQueued ? 1 : 0, exactSeqQueued ? 1 : 0,
                    *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_LOCAL_SYNC_APPLIED) ? 1 : 0,
                    *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_IS_UPDATING_UI) ? 1 : 0);
        }
        // If Jawaker still owns the exact optimistic PendingMove, sending a
        // second action would stack input over an unresolved server request.
        // Let SyncFromGameStore remove/reconcile that sequence first.
        if (trackedQueueOutstanding &&
            !solitaire_liveness_policy::queueWaitMustEnd(
                true, flightAge, SOL_ACTION_HARD_TIMEOUT_MS)) return true;
        if (flightAge < SOL_ACTION_HARD_TIMEOUT_MS) return true;

        // A hard no-change timeout is the only timer-driven rejection. Restore
        // Talon bookkeeping before the planner considers another root edge.
        g_actualRecycleCount = g_pendingRecycleCountBefore;
        g_stagnantRecycleCount = g_pendingStagnantRecycleCountBefore;
        g_pacingBackoffMs = std::min(
            SOL_MAX_PACING_BACKOFF_MS,
            g_pacingBackoffMs + SOL_PACING_BACKOFF_STEP_MS);
        registerPhysicalRootFailure(g_pendingRootKey, g_pendingMoveKey);
        if (rootKey != g_pendingRootKey)
            registerPhysicalRootFailure(rootKey, g_pendingMoveKey);
        SOL_LOG("[ACTION_TIMEOUT] root=%s move=%s age=%llu seq=%lld pendingNow=%d queuedNow=%d pacing=%llu",
                g_pendingRootKey.c_str(), g_pendingMoveKey.c_str(),
                (unsigned long long)flightAge,
                (long long)g_pendingSequence,
                pendingNow,
                pendingMovesContainSequence(localGs, g_pendingSequence) ? 1 : 0,
                (unsigned long long)g_pacingBackoffMs);
        g_waitingForMoveResult = false;
        g_committedPlan.clear();
        g_committedPlanAllowsRecoveryPreparation = false;
        g_committedPlanIsVerifiedWin = false;
        // Steps queued behind a rejected action were planned from its result.
        discardChainedSteps();
        discardPendingSubmissionEffects();
        // expertState was built with the speculative recycle count. Re-read on
        // the next tick before recording history or dispatching another move.
        return true;
    }

    if (g_pendingObservedPhysicalKey != physicalKey) {
        g_pendingObservedPhysicalKey = physicalKey;
        g_pendingObservedSinceMs = now;
        if (flightAge < SOL_ACTION_HARD_TIMEOUT_MS) return true;
    }
    const bool uiUpdating =
        *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_IS_UPDATING_UI) != 0;
    // Fast acknowledgement is allowed only after two matching observations,
    // exact logical AND physical outcomes, the tracked server queue retired,
    // and Unity explicitly reports it is no longer updating the UI. Keep the
    // conservative settling window for every optimistic/mismatched snapshot.
    const uint64_t stableMs = phoneFastUiProfile() && g_pendingWasQueued && g_pendingSequence >= 0 &&
        !trackedQueueOutstanding && pendingNow == 0 &&
        localSequenceNumber(localGs) == g_pendingSequence && !uiUpdating &&
        physicalKey == g_pendingPhysicalExpectedKey && rootKey == g_pendingExpectedKey
        ? 2 : SOL_POST_ACK_STABLE_MS;
    if (flightAge < SOL_ACTION_HARD_TIMEOUT_MS &&
        now - g_pendingObservedSinceMs < stableMs) return true;

    const bool exactExpectedState =
        physicalKey == g_pendingPhysicalExpectedKey &&
        rootKey == g_pendingExpectedKey;
    const bool optimisticPipelineAck =
        solitaire_liveness_policy::canOptimisticallySettle({
            phoneFastUiProfile(),
            trackedQueueOutstanding && g_pendingWasQueued &&
                g_pendingSequence >= 0,
            localSequenceNumber(localGs) == g_pendingSequence,
            uiUpdating,
            exactExpectedState,
            pendingNow,
            g_pendingCountAtSubmit,
            flightAge,
        });

    // The expected LocalGs board may be optimistic. For a sequence we observed
    // entering PendingMoves, require the game's backend reconciliation to retire
    // that exact entry before calling the action settled.
    if (trackedQueueOutstanding && !optimisticPipelineAck &&
        !solitaire_liveness_policy::queueWaitMustEnd(
            true, flightAge, SOL_ACTION_HARD_TIMEOUT_MS)) {
        if (flightAge > 1000 && now - g_lastDiagMs > 1000) {
            g_lastDiagMs = now;
            SOL_LOG("[PENDING_QUEUE] move=%s age=%llu seq=%lld current=%lld pending=%d ui=%d",
                    g_pendingMoveKey.c_str(), (unsigned long long)flightAge,
                    (long long)g_pendingSequence,
                    (long long)localSequenceNumber(localGs),
                    pendingNow,
                    *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_IS_UPDATING_UI) ? 1 : 0);
        }
        return true;
    }
    // The tick now observes during the animation, so wait here (before the
    // one-shot settle logs) rather than logging the same ACK every frame.
    if (uiUpdating && flightAge < effectiveUiWatchdogMs()) return true;
    if (optimisticPipelineAck) {
        SOL_LOG("[PIPELINE_ACK] move=%s age=%llu pending=%d/%d exact=1",
                g_pendingMoveKey.c_str(), (unsigned long long)flightAge,
                pendingNow, SOL_OPTIMISTIC_PIPELINE_MAX);
    } else if (trackedQueueOutstanding) {
        SOL_LOG("[PENDING_WATCHDOG] move=%s age=%llu pending=%d exact=%d",
                g_pendingMoveKey.c_str(), (unsigned long long)flightAge,
                pendingNow, exactExpectedState ? 1 : 0);
    }

    g_waitingForMoveResult = false;
    // With steps queued behind this flight the server may already have
    // applied some of them too; that board also verifies this flight.
    const int chainHit = chainedStepMatching(physicalKey);
    if (physicalKey == g_pendingPhysicalExpectedKey || chainHit >= 0) {
        const uint64_t recoveryStep = pacingRecoveryStepMs();
        g_pacingBackoffMs = g_pacingBackoffMs > recoveryStep
            ? g_pacingBackoffMs - recoveryStep
            : 0;
        SOL_LOG("[ACTION_SETTLED] move=%s age=%llu stable=%llu pacing=%llu alsoSettled=%d",
                g_pendingMoveKey.c_str(),
                (unsigned long long)flightAge,
                (unsigned long long)(now - g_pendingObservedSinceMs),
                (unsigned long long)g_pacingBackoffMs, chainHit + 1);
        commitPendingSubmissionEffects();
        // Compares rootKey with the last verified prediction (clearing the
        // unsent plan on a mismatch, as before) and promotes any next step.
        return settleChainedSteps(chainHit, rootKey);
    } else {
        SOL_LOG("[ACTION_INTERRUPTED] move=%s age=%llu stable=%llu",
                g_pendingMoveKey.c_str(),
                (unsigned long long)flightAge,
                (unsigned long long)(now - g_pendingObservedSinceMs));
        g_committedPlan.clear();
        g_committedPlanAllowsRecoveryPreparation = false;
        g_committedPlanIsVerifiedWin = false;
        g_actualRecycleCount = g_pendingRecycleCountBefore;
        g_stagnantRecycleCount = g_pendingStagnantRecycleCountBefore;
        // An unexpected board invalidates everything queued behind this flight.
        discardChainedSteps();
        discardPendingSubmissionEffects();
        // Rebuild the GameState with reconciled bookkeeping next tick.
        return true;
    }
}

static bool chooseAndExecuteExpertMove(void* localGs, const GameState& state) {
    g_accessStage.store(40, std::memory_order_release);
    g_lastDispatchOutcome = DispatchOutcome::Rejected;
    const uint64_t now = nowMs();
    const std::string rootKey = virtualPositionKey(state);
    const std::string physicalKey = physicalPositionKey(state);

    // Tick reconciles pending flights before admitting this state to strategic
    // history. Keep a defensive gate in case this helper is ever called alone.
    if (g_waitingForMoveResult) return false;
    if (now < g_pipelineDrainUntilMs) {
        g_dispatchDeferredThisTick = true;
        return false;
    }
    if (g_committedPlanIsVerifiedWin && !committedWinMatches(state)) {
        g_committedPlan.clear();
        g_committedPlanIsVerifiedWin = false;
        g_committedWinExpectedKey.clear();
    }

    if (!g_dispatchRetryPhysicalKey.empty()) {
        if (physicalKey != g_dispatchRetryPhysicalKey ||
            now >= g_dispatchRetryAtMs) {
            g_dispatchRetryPhysicalKey.clear();
            g_dispatchRetryAtMs = 0;
        } else {
            g_dispatchDeferredThisTick = true;
            return false;
        }
    }

    // A complete winning certificate always outranks the local stock router.
    // Accept route-tail work as soon as the preceding physical action is
    // acknowledged, before considering any liveness shortcut.
    if (g_committedPlan.empty() && takeAnticipatorySearch(state, rootKey) &&
        g_persistentPlanner.hasWinningCertificate) {
        const auto& win = g_persistentPlanner.winningCertificate;
        if (verifiedWinningLine(state, win.line)) {
            g_committedPlan = win.line;
            g_committedPlanScore = win.score;
            g_committedPlanAllowsRecoveryPreparation = false;
            g_committedPlanIsVerifiedWin = true;
            trackCommittedWinState(state);
        }
    }

    // Do this before the exhausted-search latch.  A board whose only legal
    // action is Talon does not need another wide planner rebuild: KING already
    // knows the exact shortest useful stock exit and can stream that route
    // through the ordinary ACK gate without waiting for a manual wake-up tap.
    if (g_committedPlan.empty() && !g_committedPlanIsVerifiedWin) {
        std::vector<Move> kingStockRoute;
        int paidRecycles = 0;
        if (findKingStockExitLine(state, kingStockRoute, &paidRecycles)) {
            g_committedPlan = std::move(kingStockRoute);
            g_committedPlanScore = 900000 -
                paidRecycles * SOL_CHARGED_RECYCLE_SEARCH_COST;
            g_committedPlanAllowsRecoveryPreparation = false;
            g_committedPlanIsVerifiedWin = false;
            g_committedWinExpectedKey.clear();
            SOL_LOG("[KING_STOCK_ROUTE] steps=%d paidRecycles=%d score=%d",
                    (int)g_committedPlan.size(), paidRecycles,
                    g_committedPlanScore);
        }
    }

    // A fully exhausted state is parked briefly, not forever. The emergency
    // path remains active during the cooldown; afterwards the deepest bounded
    // recovery pass is reopened. This prevents a solvable board from becoming
    // permanently idle because one finite search slice ran out of branches.
    if (!g_exhaustedRecoveryKey.empty() &&
        rootKey == g_exhaustedRecoveryKey && !g_committedPlan.empty()) {
        g_exhaustedRecoveryKey.clear();
        g_exhaustedRecoveryAt = 0;
    }
    if (!g_exhaustedRecoveryKey.empty()) {
        if (rootKey == g_exhaustedRecoveryKey) {
            // Complete knowledge cannot improve through another mapping
            // burst, and a maximum-escalation empty frontier will rebuild the
            // same failed search every 650 ms. A real board/knowledge change
            // clears this exact-key latch in observeStrategicProgress().
            if (g_committedPlan.empty() && knownCardCount(state) == 52 &&
                !hasOpaqueCards(state) &&
                g_recoveryEscalationByState[rootKey] >=
                    SOL_MAX_RECOVERY_ESCALATION) return false;
            if (now - g_exhaustedRecoveryAt < SOL_EXHAUSTED_RETRY_MS) return false;
            // hydratePerfectInformation() runs before this function. Give an
            // armed mapping-recovery burst all of its bounded native sync
            // attempts before rebuilding the same planner frontier.
            if (!g_mappingValidated &&
                g_mappingSyncAttempts < SOL_MAPPING_SYNC_MAX_ATTEMPTS) {
                return false;
            }
            // When both search and all mapping bursts are exhausted, remain
            // latched until the exact state changes. This removes the captured
            // 650 ms recovery-search-exhausted CPU/log loop.
            if (!g_mappingValidated &&
                g_mappingRecoveryBursts >= SOL_MAPPING_SYNC_RECOVERY_BURSTS &&
                g_mappingSyncAttempts >= SOL_MAPPING_SYNC_MAX_ATTEMPTS) {
                return false;
            }
            g_exhaustedRecoveryKey.clear();
            g_exhaustedRecoveryAt = 0;
            g_recoveryMode = true;
            g_recoveryCheckpointId = 0;
            g_lastRecoveryAt = now;
            g_progressTracker.recoveryAttemptsAtState = 0;
            g_progressTracker.stallPending = false;
            g_progressTracker.stallReason.clear();
            SOL_LOG("[RECOVERY] exhausted-state cooldown elapsed; reopening bounded search escalation=%d",
                    g_recoveryEscalationByState[rootKey]);
        } else {
            g_exhaustedRecoveryKey.clear();
            g_exhaustedRecoveryAt = 0;
        }
    }

    detectPlannerIdleStall(rootKey, now);
    if (g_progressTracker.stallPending) {
        // A certified route can contain more than one preparation before its
        // reveal/Foundation milestone. Re-certify the whole observed remainder
        // instead of judging only its next step and throwing away a valid path.
        CertifiedPlanPrefix remaining;
        const bool remainingProvesWin = committedWinMatches(state);
        if (!remainingProvesWin && !g_committedPlan.empty()) {
            remaining = certifyPlanPrefix(
                state, g_committedPlan, false, true,
                g_committedPlanAllowsRecoveryPreparation);
        }
        if (remainingProvesWin || remaining.valid) {
            if (!remainingProvesWin) {
                const size_t keep = (size_t)remaining.progressIndex + 1;
                if (keep < g_committedPlan.size()) g_committedPlan.resize(keep);
                g_committedPlanIsVerifiedWin = false;
            }
            g_progressTracker.stallPending = false;
            g_progressTracker.stallReason.clear();
            SOL_LOG("[PLAN_GUARD] retained committed remainder=%d milestone=%s",
                    (int)g_committedPlan.size(),
                    remainingProvesWin ? "verified-win" : remaining.reason.c_str());
        } else {
            g_committedPlanIsVerifiedWin = false;
            activateStrategicRecovery(state, rootKey);
        }
    }

    // If the current board itself admits a monotone Tableau-to-Foundation
    // completion, construct that exact line in one pass. The independent
    // verifier and ordinary physical ACK still guard every dispatched card.
    // The game's own auto-complete request may finish earlier; this is the
    // legal fallback when its separate GameStore eligibility bit is absent.
    if (g_committedPlan.empty() && autoCompleteLayoutCandidate(state) &&
        completeKnownDeck(state) && !hasOpaqueCards(state)) {
        std::vector<Move> directFinish;
        if (provenAutoCompleteLayout(state, &directFinish) &&
            verifiedWinningLine(state, directFinish)) {
            g_committedPlan = std::move(directFinish);
            g_committedPlanScore = 2000000;
            g_committedPlanAllowsRecoveryPreparation = false;
            g_committedPlanIsVerifiedWin = true;
            trackCommittedWinState(state);
            g_foundationFinishAttemptedKey = rootKey;
            g_foundationFinishProbe = FoundationFinishProbe{};
            SOL_LOG("[ENDGAME] direct certified finish steps=%d",
                    (int)g_committedPlan.size());
        }
    }

    // Low Foundation cards are timing-critical in Draw-3: one extra Talon
    // click hides them again and the ordinary planner may then spend a whole
    // orbit proving a different reveal. Install the exact shortest route now,
    // but keep every physical step behind the same legality and ACK gates.
    // A verified winning certificate is the only route this fast lane cannot
    // replace.
    const bool victoryOracleEndgame = completeKnownDeck(state) &&
        (hiddenCardCount(state) <= SOL_VICTORY_ENDGAME_HIDDEN ||
         virtualFoundationCount(state) >= SOL_VICTORY_ENDGAME_FOUNDATION);
    if (!g_committedPlanIsVerifiedWin && !victoryOracleEndgame) {
        std::vector<Move> priorityFoundation;
        if (findPriorityStockFoundationLine(state, priorityFoundation)) {
            bool changed = priorityFoundation.size() != g_committedPlan.size();
            for (size_t i = 0; !changed && i < priorityFoundation.size(); ++i) {
                const Move& a = priorityFoundation[i];
                const Move& b = g_committedPlan[i];
                changed = a.kind != b.kind || a.from != b.from ||
                          a.to != b.to || a.start != b.start;
            }
            g_committedPlan = std::move(priorityFoundation);
            g_committedPlanScore = 750000;
            g_committedPlanAllowsRecoveryPreparation = false;
            g_committedPlanIsVerifiedWin = false;
            g_committedWinExpectedKey.clear();
            if (changed) {
                SOL_LOG("[STOCK_FAST_LANE] steps=%d terminal=foundation",
                        (int)g_committedPlan.size());
            }
        }
    }

    // A completely known, short endgame may have a forced Foundation-only
    // completion. Prove it in small resumable slices, then execute through
    // the ordinary physical ACK/legality path. A slow phone can no longer
    // lose this opportunity because one proof slice hit a clock deadline.
    if (g_committedPlan.empty() &&
        g_foundationFinishAttemptedKey != rootKey &&
        knownCardCount(state) == 52 &&
        hiddenCardCount(state) <= SOL_VICTORY_ENDGAME_HIDDEN) {
        if (g_foundationFinishProbe.rootKey != rootKey) {
            g_foundationFinishProbe = FoundationFinishProbe{};
            g_foundationFinishProbe.rootKey = rootKey;
            g_foundationFinishProbe.cursor = state;
            g_foundationFinishProbe.line.reserve(96);
            g_foundationFinishProbe.visited.reserve(192);
        }
        const FoundationFinishResult proof = advanceFoundationOnlyFinishPlan(
            g_foundationFinishProbe, 48, nowMs() + 4);
        if (proof == FoundationFinishResult::Pending) {
            g_dispatchDeferredThisTick = true;
            return false;
        }
        g_foundationFinishAttemptedKey = rootKey;
        if (proof == FoundationFinishResult::Won &&
            verifiedWinningLine(state, g_foundationFinishProbe.line)) {
            g_committedPlan = std::move(g_foundationFinishProbe.line);
            g_committedPlanScore = 2000000;
            g_committedPlanAllowsRecoveryPreparation = false;
            g_committedPlanIsVerifiedWin = true;
            trackCommittedWinState(state);
            SOL_LOG("[ENDGAME] verified foundation-only finish steps=%d",
                    (int)g_committedPlan.size());
        }
        g_foundationFinishProbe = FoundationFinishProbe{};
    }

    // Continue the verified multi-step route found at the previous root.  A
    // plan is committed only up to its first proven progress milestone; every
    // physical step is still validated against the freshly-read LocalGs.
    if (!g_committedPlan.empty()) {
        const bool certifiedWin = committedWinMatches(state);
        const Move nextMove = g_committedPlan.front();
        GameState expected = state;
        const int recycleCountBefore = g_actualRecycleCount;
        const int stagnantRecycleCountBefore = g_stagnantRecycleCount;
        const bool expectedReady = applyVirtualMove(expected, nextMove);
        const bool proofSafe = certifiedWin || !expectedReady ||
            !completeKnownDeck(expected) || !provenDependencyDeadEnd(expected);
        if ((certifiedWin || physicalCandidateAllowed(state, nextMove, now)) &&
            expectedReady && proofSafe &&
            (certifiedWin || g_deadStateCache.find(solverPositionKey(expected)) == g_deadStateCache.end()) &&
            executeExpertMove(localGs, state, nextMove, g_committedPlanScore)) {
            const uint64_t checkpointId = saveDecisionCheckpoint(
                state, nextMove, g_committedPlan,
                legalAlternativeKeys(state, nextMove),
                g_committedPlanScore);
            g_committedPlan.erase(g_committedPlan.begin());
            if (g_committedPlan.empty()) g_committedPlanIsVerifiedWin = false;
            trackCommittedWinState(expected);
            g_pendingRootKey = rootKey;
            g_pendingExpectedKey = virtualPositionKey(expected);
            g_pendingPhysicalRootKey = physicalKey;
            g_pendingPhysicalExpectedKey = physicalPositionKey(expected);
            g_pendingMoveKey = moveKey(nextMove);
            g_pendingMoveAt = now;
            g_pendingRecycleCountBefore = recycleCountBefore;
            g_pendingStagnantRecycleCountBefore = stagnantRecycleCountBefore;
            armPendingFlightObservation();
            g_pendingCheckpointId = checkpointId;
            g_pendingExpectedState = expected;
            g_pendingChainable = true;
            g_pendingFlipsTableau = hiddenCardCount(expected) < hiddenCardCount(state);
            g_waitingForMoveResult = true;
            scheduleAnticipatorySearch(expected);
            SOL_LOG("expert committed move=%s remaining=%d score=%d",
                    g_pendingMoveKey.c_str(), (int)g_committedPlan.size(),
                    g_committedPlanScore);
            return true;
        }
        if (deferPhysicalDispatch(state)) return false;
        SOL_LOG("expert committed route invalidated remaining=%d proofSafe=%d: replanning",
                (int)g_committedPlan.size(), proofSafe ? 1 : 0);
        g_committedPlan.clear();
        g_committedPlanIsVerifiedWin = false;
    }

    // Card layout keys intentionally ignore passes for loop detection. Search
    // values do not: a newly spent free pass changes every future recycle's
    // score cost even when a complete stock rotation returns to this layout.
    if (g_persistentPlanner.rootKey != rootKey ||
        g_persistentPlanner.rootRecycles != state.recycles)
        initializePersistentRoot(state, rootKey);
    advancePersistentSearch(state);
    if (g_persistentPlanner.capacityReached && g_persistentPlanner.roots.empty() &&
        g_persistentPlanner.recoveryRoots.empty()) {
        // A bounded arena running out of room is not evidence of an unsolvable
        // deal. Hand it to the existing bounded recovery policy, not an idle
        // frontier that can never expand or a global dead-state certificate.
        clearPlannerFrontier();
        flagStrategicStall("search-resource-limit");
        activateStrategicRecovery(state, rootKey);
        return false;
    }
    widenStalledProofSearch(state, rootKey, nowMs());
    SOL_TRACE("[SEARCH] primary-return frame=%d roots=%d frontier=%d",
              g_persistentPlanner.frames, (int)g_persistentPlanner.roots.size(),
              (int)g_persistentPlanner.frontier.size());
    // Once speculative bridges have spent their per-epoch allowance, retain
    // the same search engine for a deeper bounded proof. Do not replace a
    // visible Tableau loop with an unbounded Unity-thread search loop.
    if (g_recoveryMode &&
        g_progressTracker.recoveryPreparations >= SOL_MAX_RECOVERY_PREPARATIONS &&
        g_persistentPlanner.roots.empty() &&
        (g_persistentPlanner.frames >= SOL_RECOVERY_PROOF_ONLY_FRAMES ||
         g_persistentPlanner.expanded >= 30000)) {
        SOL_LOG("[RECOVERY] proof-only budget exhausted frames=%d expanded=%d prep=%d/%d",
                g_persistentPlanner.frames, g_persistentPlanner.expanded,
                g_progressTracker.recoveryPreparations,
                SOL_MAX_RECOVERY_PREPARATIONS);
        clearPlannerFrontier();
        flagStrategicStall("recovery-proof-budget-exhausted");
        activateStrategicRecovery(state, rootKey);
        return false;
    }
    detectPlannerIdleStall(rootKey, now);
    if (g_progressTracker.stallPending &&
        activateStrategicRecovery(state, rootKey)) {
        // A retained current-root frontier remains valid; give its newly
        // recovery-weighted siblings one bounded slice immediately.
        advancePersistentSearch(state);
    }

    const int requiredHorizon = requiredRootHorizon(g_persistentPlanner);
    const int bestHorizon = bestExecutableRootHorizon(rootKey);
    const bool horizonReady = !g_persistentPlanner.perfectInformation ||
                              bestHorizon >= requiredHorizon;
    const bool normalBudgetReady = !g_persistentPlanner.endgame &&
        g_persistentPlanner.frames >= SOL_NORMAL_SEARCH_FRAMES &&
        g_persistentPlanner.expanded >= 3000;
    const bool endgameBudgetReady = g_persistentPlanner.endgame &&
        g_persistentPlanner.frames >= SOL_ENDGAME_SEARCH_FRAMES + 1 &&
        g_persistentPlanner.expanded >= 8000;
    bool obviousProgressReady = false;
    if (g_persistentPlanner.frames >= 1 &&
        g_persistentPlanner.expanded >= 600) {
        for (const auto& entry : g_persistentPlanner.roots) {
            const RootPlan& plan = entry.second;
            const int rootHorizon = rootHorizonFor(plan.first);
            if (plan.line.size() <= 2 &&
                immediateMoveTier(state, plan.first) >= 90 &&
                (!g_persistentPlanner.perfectInformation ||
                 (!g_persistentPlanner.endgame && rootHorizon >= 8))) {
                obviousProgressReady = true;
                break;
            }
        }
    }
    // Full-deal information earns a short survival proof before execution.
    // A bounded fallback prevents the proof gate itself from looking like a
    // freeze on a wide early-game tree.
    // Never time out of a full-information endgame into a merely attractive
    // local move.  Continue until a verified win, the required survival
    // horizon, or a genuinely exhausted/resource-bounded frontier. Earlier
    // full-information roots retain a much later bounded fallback so play
    // stays live on unusually wide openings.
    const bool boundedProofFallback = !g_persistentPlanner.perfectInformation
        ? (!g_persistentPlanner.endgame
            ? (g_persistentPlanner.frames >= 8 || g_persistentPlanner.expanded >= 5000)
            : (g_persistentPlanner.frames >= 13 || g_persistentPlanner.expanded >= 12000))
        : (!g_persistentPlanner.endgame && bestHorizon >= 12 &&
           (g_persistentPlanner.frames >= 16 || g_persistentPlanner.expanded >= 16000));
    const bool ready = g_persistentPlanner.solved ||
                       (g_persistentPlanner.capacityReached && g_persistentPlanner.foundProgress) ||
                       (g_persistentPlanner.foundProgress && obviousProgressReady) ||
                       (g_persistentPlanner.foundProgress &&
                        (((normalBudgetReady || endgameBudgetReady) && horizonReady) ||
                          boundedProofFallback)) ||
                       g_persistentPlanner.frontier.empty();
    const bool recoveryReady = g_recoveryMode && g_persistentPlanner.roots.empty() &&
                               g_progressTracker.recoveryPreparations <
                                   SOL_MAX_RECOVERY_PREPARATIONS &&
                               !g_persistentPlanner.recoveryRoots.empty() &&
                               (g_persistentPlanner.frontier.empty() ||
                                (!g_persistentPlanner.endgame &&
                                 (g_persistentPlanner.frames >= 18 ||
                                  g_persistentPlanner.expanded >= 9000)) ||
                                (g_persistentPlanner.endgame &&
                                 (g_persistentPlanner.frames >= 36 ||
                                  g_persistentPlanner.expanded >= 18000)));

    enum class RootSource { Normal, Recovery };
    RootSource rootSource = RootSource::Normal;
    std::vector<RootPlan> alternatives;
    if (g_persistentPlanner.hasWinningCertificate &&
        verifiedWinningLine(state, g_persistentPlanner.winningCertificate.line)) {
        alternatives.push_back(g_persistentPlanner.winningCertificate);
    } else if (ready && !g_persistentPlanner.roots.empty()) {
        alternatives.reserve(g_persistentPlanner.roots.size());
        for (const auto& entry : g_persistentPlanner.roots)
            alternatives.push_back(entry.second);
    } else if (recoveryReady) {
        rootSource = RootSource::Recovery;
        alternatives.reserve(g_persistentPlanner.recoveryRoots.size());
        for (const auto& entry : g_persistentPlanner.recoveryRoots)
            alternatives.push_back(entry.second);
    } else {
        if (g_persistentPlanner.frontier.empty()) {
            flagStrategicStall(g_recoveryMode ? "recovery-search-exhausted" :
                                                "planner-frontier-exhausted");
            activateStrategicRecovery(state, rootKey);
        }
        return false;
    }

    struct RankedPlan {
        RootPlan plan;
        int effective = -10000000;
        int horizon = 0;
        int terminalLeaves = 0;
        int localScoreDelta = 0;
        int prefixActions = 0;
        int prefixTalonActions = 0;
        int prefixPaidRecycles = 0;
    };
    std::vector<RankedPlan> ranked;
    ranked.reserve(alternatives.size());
    for (const RootPlan& plan : alternatives) {
        if (rootMoveFailed(rootKey, plan.first)) continue;
        const int horizon = rootHorizonFor(plan.first);
        const int terminalLeaves = rootTerminalLeavesFor(plan.first);
        const PlanTempoValue tempo = valueCertifiedPrefix(state, plan.line);
        int effective = plan.score + immediateMoveTier(state, plan.first) * 90 -
                        (int)plan.line.size() * 12 -
                        strategicLinePenalty(state, plan.line);
        const int horizonWeight = g_persistentPlanner.endgame ? 2600 : 1700;
        const int terminalWeight = g_persistentPlanner.endgame ? 1500 : 900;
        effective += std::min(horizon, requiredHorizon + 16) * horizonWeight;
        effective -= std::min(terminalLeaves, 20) * terminalWeight;
        // Only near-tie scale: exact action points and fewer physical actions
        // improve final/time score, but can never outweigh a meaningful safety
        // or horizon advantage. Recovery branches keep purely strategic order.
        if (rootSource == RootSource::Normal) {
            effective += std::clamp(tempo.localScoreDelta, -160, 200) * 5;
            effective -= tempo.actions * 55 +
                         tempo.talonActions * SOL_TALON_STEP_SEARCH_COST +
                         tempo.paidRecycles * SOL_CHARGED_RECYCLE_SEARCH_COST;
        }
        if (g_persistentPlanner.perfectInformation &&
            horizon < requiredHorizon && !g_persistentPlanner.frontier.empty())
            effective -= (requiredHorizon - horizon) * 9000;
        ranked.push_back({plan, effective, horizon, terminalLeaves,
                          tempo.localScoreDelta, tempo.actions,
                          tempo.talonActions, tempo.paidRecycles});
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedPlan& a,
                                                const RankedPlan& b) {
        if (a.plan.wins != b.plan.wins) return a.plan.wins;
        if (a.effective != b.effective) return a.effective > b.effective;
        if (a.plan.line.size() != b.plan.line.size())
            return a.plan.line.size() < b.plan.line.size();
        return a.horizon > b.horizon;
    });

    for (const RankedPlan& rankedPlan : ranked) {
        const RootPlan& plan = rankedPlan.plan;
        const bool lineProvesWin = plan.wins && verifiedWinningLine(state, plan.line);
        if (!lineProvesWin && !physicalCandidateAllowed(state, plan.first, now)) {
            g_failedStrategicBranches[rootKey].insert(moveKey(plan.first));
            discardPlannerRootEdge(rootKey, moveKey(plan.first));
            continue;
        }
        CertifiedPlanPrefix certified;
        if (lineProvesWin) {
            certified.valid = true;
            certified.progressIndex = (int)plan.line.size() - 1;
            certified.reason = "verified-win";
        } else {
            certified = certifyPlanPrefix(state, plan.line, false, true,
                                          rootSource == RootSource::Recovery);
        }
        if (!certified.valid) {
            if (certified.failureIndex == 0) {
                // Every backup begins with the same impossible first action.
                // Swapping backups here caused the 601-point case to reject
                // thousands of Talon plans while growing a useless frontier.
                const std::string firstKey = moveKey(plan.first);
                g_persistentPlanner.uncertifiableFirstEdges.insert(firstKey);
                discardPlannerRootEdge(rootKey, firstKey);
                SOL_LOG("[PLAN_GUARD] rejected-first move=%s reason=%s",
                        firstKey.c_str(), certified.reason.c_str());
                continue;
            }
            // Do not blacklist the whole first edge for a bad *deeper* route;
            // another continuation below the same legal root may still win.
            // Promote the already-searched runner-up when present; only remove
            // this source's published line when no certified sibling remains.
            const bool promotedBackup = promotePlannerRootBackup(
                rootKey, moveKey(plan.first),
                rootSource == RootSource::Recovery);
            if (!certified.failureStateKey.empty()) {
                int& failures = g_strategicStateFailures[certified.failureStateKey];
                failures = std::min(5, failures + 1);
            }
            SOL_LOG("[PLAN_GUARD] rejected move=%s reason=%s step=%d line=%d backup=%d",
                    moveKey(plan.first).c_str(), certified.reason.c_str(),
                    certified.failureIndex, (int)plan.line.size(),
                    promotedBackup ? 1 : 0);
            // The ranked snapshot still contains the rejected primary. Let the
            // next decision tick rank the promoted sibling against all roots.
            if (promotedBackup) return false;
            continue;
        }
        std::vector<Move> executableLine(
            plan.line.begin(),
            lineProvesWin ? plan.line.end() :
                            plan.line.begin() + certified.progressIndex + 1);
        // Certification used to prove only that the prefix reached a local
        // milestone.  With the complete deal known, replay the entire prefix
        // and run the sound dependency closure before any physical click.  A
        // reveal or score gain that leaves no possible completion is rejected
        // here, while all sibling roots remain available to the same planner.
        if (!lineProvesWin && g_persistentPlanner.perfectInformation) {
            GameState certifiedEnd = state;
            bool replayed = true;
            for (const Move& step : executableLine) {
                if (!applyVirtualMove(certifiedEnd, step)) {
                    replayed = false;
                    break;
                }
            }
            if (!replayed || provenDependencyDeadEnd(certifiedEnd)) {
                const std::string firstKey = moveKey(plan.first);
                g_failedStrategicBranches[rootKey].insert(firstKey);
                discardPlannerRootEdge(rootKey, firstKey);
                if (replayed) g_deadStateCache.insert(solverPositionKey(certifiedEnd));
                SOL_LOG("[VICTORY_ORACLE] rejected prefix move=%s steps=%d replayed=%d reason=dependency-dead-end",
                        firstKey.c_str(), (int)executableLine.size(), replayed ? 1 : 0);
                continue;
            }
        }
        GameState expected = state;
        if (!applyVirtualMove(expected, plan.first)) continue;
        if (!lineProvesWin && g_deadStateCache.find(solverPositionKey(expected)) != g_deadStateCache.end()) continue;
        const int recycleCountBefore = g_actualRecycleCount;
        const int stagnantRecycleCountBefore = g_stagnantRecycleCount;
        if (!executeExpertMove(localGs, state, plan.first, plan.score)) {
            if (deferPhysicalDispatch(state)) return false;
            registerPhysicalRootFailure(rootKey, moveKey(plan.first));
            continue;
        }
        std::vector<std::string> checkpointAlternatives;
        checkpointAlternatives.reserve(ranked.size());
        for (const RankedPlan& other : ranked) {
            const std::string otherKey = moveKey(other.plan.first);
            if (otherKey != moveKey(plan.first) &&
                !rootMoveFailed(rootKey, other.plan.first))
                checkpointAlternatives.push_back(otherKey);
        }
        const uint64_t checkpointId = saveDecisionCheckpoint(
            state, plan.first, executableLine,
            std::move(checkpointAlternatives), plan.score);
        g_committedPlan.clear();
        if (executableLine.size() > 1)
            g_committedPlan.assign(executableLine.begin() + 1,
                                   executableLine.end());
        g_committedPlanScore = plan.score;
        g_committedPlanAllowsRecoveryPreparation =
            rootSource == RootSource::Recovery;
        g_committedPlanIsVerifiedWin = lineProvesWin &&
            !g_committedPlan.empty();
        trackCommittedWinState(expected);
        g_pendingRootKey = rootKey;
        g_pendingExpectedKey = virtualPositionKey(expected);
        g_pendingPhysicalRootKey = physicalKey;
        g_pendingPhysicalExpectedKey = physicalPositionKey(expected);
        g_pendingMoveKey = moveKey(plan.first);
        g_pendingMoveAt = now;
        g_pendingRecycleCountBefore = recycleCountBefore;
        g_pendingStagnantRecycleCountBefore = stagnantRecycleCountBefore;
        armPendingFlightObservation();
        g_pendingCheckpointId = checkpointId;
        g_pendingExpectedState = expected;
        g_pendingChainable = true;
        g_pendingFlipsTableau = hiddenCardCount(expected) < hiddenCardCount(state);
        g_waitingForMoveResult = true;
        scheduleAnticipatorySearch(expected);
        const bool recovered = g_recoveryMode;
        if (certified.reason == "recovery-preparation") {
            // Spend the speculative budget only if Unity actually settles the
            // first submitted action, as with emergency preparations.
            g_pendingCountsRecoveryPreparation = true;
        }
        if (recovered) {
            SOL_LOG("[RECOVERY_RESULT] selectedBranch=%s searchNodes=%d newPlanLength=%d",
                    g_pendingMoveKey.c_str(), g_persistentPlanner.expanded,
                    (int)executableLine.size());
            g_recoveryMode = false;
            g_recoveryCheckpointId = 0;
            g_recoveryGraceTransitions = 4;
        }
        g_progressTracker.stallPending = false;
        g_progressTracker.stallReason.clear();
        SOL_LOG("expert selected move=%s score=%d effective=%d line=%d/%d milestone=%s horizon=%d/%d terminal=%d localScore=%d prefixActions=%d talonActions=%d paidRecycles=%d known=%d/52 expanded=%d frames=%d backup=%d source=%d",
                g_pendingMoveKey.c_str(), plan.score, rankedPlan.effective,
                (int)executableLine.size(), (int)plan.line.size(),
                certified.reason.c_str(), rankedPlan.horizon, requiredHorizon,
                rankedPlan.terminalLeaves, rankedPlan.localScoreDelta,
                rankedPlan.prefixActions, rankedPlan.prefixTalonActions,
                 rankedPlan.prefixPaidRecycles,
                g_persistentPlanner.rootKnownCards,
                g_persistentPlanner.expanded, g_persistentPlanner.frames,
                (int)ranked.size() - 1, (int)rootSource);
        return true;
    }
    if (g_persistentPlanner.frontier.empty()) {
        flagStrategicStall("no-executable-root-alternative");
        activateStrategicRecovery(state, rootKey);
    }
    return false;
}

static bool attemptFoundationMoves(void* localGs,
                                   const std::array<std::vector<Card>, 7>& tableau,
                                   const std::vector<Card>& talon,
                                   const std::vector<Card>& waste,
                                   const std::array<std::vector<Card>, 4>& foundations) {
    // Only submit moves that satisfy the exact rules used by the original
    // SolitaireRaycaster. OnMoveCard itself queues an action, so its return
    // value is not a legal-move verdict and must never be used as one.
    // The exposed waste card is the only card that disappears on the next
    // talon draw. Always test it first, so a legal 2♠ over A♠ (and any other
    // legal foundation advance) is never skipped for another talon cycle.
    const GameState future = makeGameState(tableau, foundations, talon, waste);
    if (!waste.empty() && waste.back().faceUp()) {
        for (int f = 0; f < 4; ++f) {
            const Card* top = foundations[f].empty() ? nullptr : &foundations[f].back();
            if (!canDropOnFoundation(waste.back(), top)) continue;
            if (!foundationAdvanceIsSafe(waste.back(), foundations)) continue;
            if (!simulate_future_moves(future, {VirtualMoveKind::Foundation, -1, f, -1}, 18)) continue;
            if (moveCard(localGs, waste.back(), g_stackPrefix + "waste",
                         foundationName(f), (int)foundations[f].size())) {
                g_rrSource = 0;
                SOL_LOG("waste-foundation id=%d foundation=%d", waste.back().id, f);
                return true;
            }
        }
    }

    std::array<std::pair<std::string, Card>, 8> sources{};
    for (int i = 0; i < 7; ++i) {
        if (!tableau[i].empty() && tableau[i].back().faceUp()) sources[i] = {tableauName(i), tableau[i].back()};
    }
    // Waste was handled above with strict priority; only tableau cards remain
    // in this pass.

    for (int step = 0; step < 7; ++step) {
        const int src = (g_rrSource + step) % 7;
        if (!sources[src].second.valid()) continue;
        for (int f = 0; f < 4; ++f) {
            const Card* foundationTop = foundations[f].empty() ? nullptr : &foundations[f].back();
            if (!canDropOnFoundation(sources[src].second, foundationTop)) continue;
            if (!foundationAdvanceIsSafe(sources[src].second, foundations)) continue;
            if (!simulate_future_moves(future, {VirtualMoveKind::Foundation, src, f, -1}, 18)) continue;
            if (moveCard(localGs, sources[src].second, sources[src].first,
                         foundationName(f), (int)foundations[f].size())) {
                g_rrSource = (src + 1) % 8;
                return true;
            }
        }
    }

    return false;
}

static bool attemptTableauMoves(void* localGs,
                                const std::array<std::vector<Card>, 7>& tableau,
                                const std::vector<Card>& talon,
                                const std::vector<Card>& waste,
                                const std::array<std::vector<Card>, 4>& foundations) {
    // The original state method validates every drop.  We only offer exposed
    // top cards, preferring waste first, then rotating tableau columns to keep
    // the play natural and to expose closed cards early.
    const GameState future = makeGameState(tableau, foundations, talon, waste);
    std::array<std::pair<std::string, Card>, 8> sources{};
    if (!waste.empty()) sources[0] = {g_stackPrefix + "waste", waste.back()};
    for (int i = 0; i < 7; ++i) {
        if (!tableau[i].empty()) {
            sources[i + 1] = {tableauName(i), firstMovableTableauCard(localGs, tableau[i])};
        }
    }

    // First pass: free a closed card. This is the strongest practical move in
    // Klondike and avoids harmless rearrangements while useful cards remain.
    for (int revealOnly = 1; revealOnly >= 0; --revealOnly) {
        for (int step = 0; step < 8; ++step) {
            const int src = (g_rrSource + step) % 8;
            const auto& source = sources[src];
            if (!source.second.faceUp()) continue;
            const bool revealsClosed = src > 0 && !tableau[src - 1].empty() &&
                                       !tableau[src - 1].front().faceUp();
            if (revealOnly && !revealsClosed) continue;
            for (int dst = 0; dst < 7; ++dst) {
                if (source.first == tableauName(dst)) continue;
                const Card* destinationTop = tableau[dst].empty() ? nullptr : &tableau[dst].back();
                if (!canDropOnTableau(source.second, destinationTop)) continue;
                if (src > 0) {
                    const int start = movableCardIndex(tableau[src - 1], source.second.id);
                    if (!isStrategicTableauMove(tableau, src - 1, start, dst, source.second)) continue;
                } else if (tableau[dst].empty() && cardRank(source.second) != 12) {
                    continue;
                }
                // A non-reveal tableau move is permitted only when a one-ply
                // simulation proves it creates an additional useful move.
                if (!revealOnly && src > 0 &&
                    !improvesNextPosition(tableau, foundations, src - 1, dst, source.second.id)) continue;
                if (src > 0 && wouldRepeatRecentTableau(tableau, src - 1, dst,
                                                         source.second.id, nowMs())) continue;
                const int start = src > 0 ? movableCardIndex(tableau[src - 1], source.second.id) : -1;
                const Move candidate{VirtualMoveKind::Tableau, src > 0 ? src - 1 : -1, dst, start};
                if (!simulate_future_moves(future, candidate, 18)) continue;
                if (moveCard(localGs, source.second, source.first, tableauName(dst),
                             (int)tableau[dst].size())) {
                    g_rrSource = (src + 1) % 8;
                    return true;
                }
            }
        }
    }

    // A valid tableau drag may start in the middle of an exposed sequence,
    // not only at its first face-up card.  Use these moves only as a fallback
    // and only when simulation proves they create a new productive option.
    for (int step = 0; step < 7; ++step) {
        const int src = (g_rrSource + step) % 7;
        const int first = firstFaceUpIndex(tableau[src]);
        if (first < 0) continue;
        for (int start = first + 1; start < (int)tableau[src].size(); ++start) {
            const Card& moving = tableau[src][start];
            if (!moving.faceUp()) continue;
            for (int dst = 0; dst < 7; ++dst) {
                if (src == dst) continue;
                const Card* destinationTop = tableau[dst].empty() ? nullptr : &tableau[dst].back();
                if (!canDropOnTableau(moving, destinationTop)) continue;
                if (!isStrategicTableauMove(tableau, src, start, dst, moving)) continue;
                if (!improvesNextPosition(tableau, foundations, src, dst, moving.id)) continue;
                if (wouldRepeatRecentTableau(tableau, src, dst, moving.id, nowMs())) continue;
                if (!simulate_future_moves(future, {VirtualMoveKind::Tableau, src, dst, start}, 18)) continue;
                if (moveCard(localGs, moving, tableauName(src), tableauName(dst),
                             (int)tableau[dst].size())) {
                    g_rrSource = (src + 1) % 8;
                    SOL_LOG("substack-move id=%d from=%d to=%d", moving.id, src, dst);
                    return true;
                }
            }
        }
    }

    RescuePlan rescue;
    if (findRescuePlan(tableau, foundations, rescue) &&
        !wouldRepeatRecentTableau(tableau, rescue.from, rescue.to, rescue.cardId, nowMs())) {
        const int index = movableCardIndex(tableau[rescue.from], rescue.cardId);
        if (index >= 0 && !simulate_future_moves(future,
                {VirtualMoveKind::Tableau, rescue.from, rescue.to, index}, 18)) return false;
        if (index >= 0 && moveCard(localGs, tableau[rescue.from][index], tableauName(rescue.from),
                                   tableauName(rescue.to), (int)tableau[rescue.to].size())) {
            g_rrSource = (rescue.from + 1) % 8;
            SOL_LOG("beam-rescue id=%d from=%d to=%d score=%d", rescue.cardId,
                    rescue.from, rescue.to, rescue.score);
            return true;
        }
    }
    return false;
}

// Unity-thread only: called from the tick, never from the Android UI thread.
static void boostFrameRateIfNeeded(uint64_t now) {
    if (!g_getTargetFrameRate || !g_setTargetFrameRate) return;
    if (g_lastFrameRateCheckMs != 0 &&
        now - g_lastFrameRateCheckMs < SOL_FRAME_RATE_RECHECK_MS) return;
    g_lastFrameRateCheckMs = now;
    const int current = g_getTargetFrameRate();
    // -1 already means "platform maximum"; anything at or above the boost is
    // left alone so a user-chosen higher setting is never lowered.
    if (current < 0 || current >= SOL_BOOST_TARGET_FRAME_RATE) return;
    if (g_savedTargetFrameRate == 0) g_savedTargetFrameRate = current;
    g_setTargetFrameRate(SOL_BOOST_TARGET_FRAME_RATE);
    SOL_LOG("[FPS] targetFrameRate %d -> %d", current, SOL_BOOST_TARGET_FRAME_RATE);
}

static void restoreFrameRateIfBoosted() {
    if (g_savedTargetFrameRate == 0 || !g_setTargetFrameRate) return;
    g_setTargetFrameRate(g_savedTargetFrameRate);
    SOL_LOG("[FPS] targetFrameRate restored to %d", g_savedTargetFrameRate);
    g_savedTargetFrameRate = 0;
    g_lastFrameRateCheckMs = 0;
}

} // namespace

void solitaire_set_speed_mode(int mode) {
    const int safeMode = mode == 0 ? 0 : 1;
    g_speedMode.store(safeMode, std::memory_order_relaxed);
    SOL_LOG("[SPEED] runtime mode=%s planner=unchanged exactAck=1",
            safeMode == 0 ? "HUMAN" : "FAST");
}

int solitaire_speed_mode() {
    return g_speedMode.load(std::memory_order_relaxed);
}

void solitaire_set_performance_profile(int profile, int sliceNodes, int sliceMs,
                                       int arenaNodes, int cardSafetyMs,
                                       int talonSafetyMs) {
    // Accept only bounded planner budgets from the signed loader. Device ULTRA
    // uses a compact slice; PC TURBO gets a wider slice. We deliberately do not
    // copy the loader's raw card/talon delays: the one-millisecond floor plus
    // exact-state ACK and adaptive backoff is safer and faster on both targets.
    const int safeProfile = std::clamp(profile, 0, 3);
    const int safeNodes = std::clamp(sliceNodes, 1800, 12800);
    // APEX SAFE is a real two-millisecond planner slice in this release.
    // Physical card/talon dispatch still retains the independent exact-state
    // acknowledgement gate and the exact-state animation pacing below.
    const int safeSliceMs = std::clamp(sliceMs, 2, 16);
    const int safeArena = std::clamp(arenaNodes, 32000, 320000);
    g_performanceProfile.store(safeProfile, std::memory_order_relaxed);
    g_searchSliceNodes.store(safeNodes, std::memory_order_relaxed);
    g_searchSliceMs.store(safeSliceMs, std::memory_order_relaxed);
    g_searchArenaNodes.store(safeArena, std::memory_order_relaxed);
    SOL_LOG("[PERFORMANCE] applied profile=%d nodes=%d sliceMs=%d arena=%d physical=1/1 uiWatchdog=%llu/%llu requestedPhysical=%d/%d exactAck=1 routeTailPrefetch=%llums",
             safeProfile, safeNodes, safeSliceMs, safeArena,
             (unsigned long long)((safeProfile == 0 || safeProfile == 2)
                 ? SOL_PHONE_CARD_UI_WATCHDOG_MS : SOL_UI_UPDATE_WATCHDOG_MS),
             (unsigned long long)((safeProfile == 0 || safeProfile == 2)
                 ? SOL_PHONE_TALON_UI_WATCHDOG_MS : SOL_UI_UPDATE_WATCHDOG_MS),
              cardSafetyMs, talonSafetyMs,
              (unsigned long long)SOL_RAPTOR_PREFETCH_MS);
}

void solitaire_init(uintptr_t il2cppBase) {
    if (!il2cppBase) return;
    // Loader builds cannot assume libil2cpp exports are visible through dlsym.
    // Keep the current resolver integration around the restored strong solver.
    g_stringNew = reinterpret_cast<StringNewFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_string_new"));
    g_objectToString = reinterpret_cast<ObjectToStringFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_object_to_string"));
    g_classGetName = reinterpret_cast<ClassNameFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_class_get_name"));
    g_domainGet = reinterpret_cast<DomainGetFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_domain_get"));
    g_domainAssemblies = reinterpret_cast<DomainAssembliesFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_domain_get_assemblies"));
    g_assemblyImage = reinterpret_cast<AssemblyImageFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_assembly_get_image"));
    g_imageName = reinterpret_cast<ImageNameFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_image_get_name"));
    g_classFromName = reinterpret_cast<ClassFromNameFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_class_from_name"));
    g_staticData = reinterpret_cast<StaticDataFn>(goat_resolve_il2cpp_export(il2cppBase, "il2cpp_class_get_static_field_data"));
    g_getUi = (GetUiFn)(il2cppBase + RVA_SOL_UI_GET_INSTANCE);
    g_getStackCards = (GetStackCardsFn)(il2cppBase + RVA_SOL_GET_STACK_CARDS);
    g_moveCard = (MoveCardFn)(il2cppBase + RVA_SOL_MOVE_CARD);
    g_talonClicked = (TalonClickedFn)(il2cppBase + RVA_SOL_TALON_CLICKED);
    g_talonUi = (TalonUiFn)(il2cppBase + RVA_SOL_UI_TALON_CLICK);
    g_syncPlayerCardsMapping = (SyncPlayerCardsMappingFn)(
        il2cppBase + RVA_SOL_SYNC_PLAYER_CARDS_MAPPING);
    g_rayMove = (RayMoveFn)(il2cppBase + RVA_SOL_RAY_MOVE_CARD);
    g_getMyIndex = (GetMyIndexFn)(il2cppBase + RVA_GAMESTORE_GET_MY_INDEX);
    g_getTargetFrameRate = (GetTargetFrameRateFn)(il2cppBase + RVA_APP_GET_TARGET_FRAME_RATE);
    g_setTargetFrameRate = (SetTargetFrameRateFn)(il2cppBase + RVA_APP_SET_TARGET_FRAME_RATE);
    static const bool planGuardSelfTestsPassed = runPlanGuardSelfTests();
    static const bool apexScoreSelfTestsPassed = runApexScoreSelfTests();
    SOL_LOG("init str=%p ui=%p stacks=%p talon=%p talonUi=%p mapSync=%p rayMove=%p domain=%p assemblies=%p image=%p class=%p static=%p",
            g_stringNew, g_getUi, g_getStackCards, g_talonClicked, g_talonUi,
            g_syncPlayerCardsMapping, g_rayMove,
            g_domainGet, g_domainAssemblies, g_assemblyImage, g_classFromName, g_staticData);
    if (!planGuardSelfTestsPassed)
        SOL_LOG("[PLAN_GUARD_TEST] FAILED: candidate guard must remain conservative");
    if (!apexScoreSelfTestsPassed)
        SOL_LOG("[APEX_SCORE_TEST] FAILED: exact score tie-break disabled by regression");
}

void solitaire_capture_ui(void* solitaireUi) {
    if (!tryEnterSolitaire()) return;
    SolitaireAccessGuard guard;
    if (solitaireUi && g_ui != solitaireUi) {
        void* newGs = *(void**)((uint8_t*)solitaireUi + OFF_SOL_UI_LOCAL_GS);
        const int64_t newStart = newGs
            ? *(const int64_t*)((const uint8_t*)newGs + OFF_SOL_START_TIME) : 0;
        const bool sameRound = newGs && newGs == g_lastLocalGs &&
            (!newStart || !g_lastRoundStartTime ||
             newStart == g_lastRoundStartTime);
        const int roundRecycles = g_actualRecycleCount;
        g_ui = solitaireUi;
        if (!sameRound) g_lastLocalGs = nullptr;
        resetRoundState(true, !sameRound);
        if (sameRound) g_actualRecycleCount = roundRecycles;
        SOL_LOG("captured UI=%p", g_ui);
    }
}

void solitaire_set_enabled(bool enabled) {
    if (enabled && !license_security_allows_game(GOAT_GAME_SOLITAIRE)) enabled = false;
    const bool previous = g_enabled.exchange(enabled, std::memory_order_acq_rel);
    if (previous != enabled) g_enableResetPending.store(true, std::memory_order_release);
    SOL_LOG("auto=%d previous=%d resetPending=%d", enabled ? 1 : 0,
            previous ? 1 : 0,
            g_enableResetPending.load(std::memory_order_relaxed) ? 1 : 0);
}

bool solitaire_is_enabled() { return g_enabled.load(std::memory_order_acquire); }

bool solitaire_is_ready() {
    return g_ui && g_stringNew && g_getStackCards && g_talonUi && g_rayMove;
}

void solitaire_probe_heartbeat(int source) {
#if GOAT_SOLITAIRE_TRACE
    if (source < 0 || source > 2) return;
    static std::atomic<uint64_t> lastAt[3]{};
    const uint64_t now = nowMs();
    uint64_t previous = lastAt[source].load(std::memory_order_relaxed);
    if (now - previous >= 1000 &&
        lastAt[source].compare_exchange_strong(previous, now,
                                                std::memory_order_relaxed)) {
        const uint64_t acquired = g_accessAcquiredMs.load(std::memory_order_acquire);
        SOL_TRACE("[HEARTBEAT] source=%d enabled=%d ui=%p reset=%d tid=%d owner=%d age=%llu stage=%d",
                source, g_enabled.load(std::memory_order_relaxed) ? 1 : 0,
                g_ui, g_enableResetPending.load(std::memory_order_relaxed) ? 1 : 0,
                gettid(), g_accessOwnerTid.load(std::memory_order_acquire),
                (unsigned long long)(acquired ? now - acquired : 0),
                g_accessStage.load(std::memory_order_acquire));
    }
#else
    (void)source;
#endif
}

static void solitaire_tick_impl() {
    if (!g_enabled.load(std::memory_order_acquire)) {
        restoreFrameRateIfBoosted();
        return;
    }
    // The solver itself is gated, not only its UI toggle. A patched button or
    // stale memory flag therefore cannot run Auto Solitaire without a current
    // signed server lease.
    if (!license_security_allows_game(GOAT_GAME_SOLITAIRE)) {
        g_enabled.store(false, std::memory_order_release);
        restoreFrameRateIfBoosted();
        return;
    }
    if (!tryEnterSolitaire()) return;
    SolitaireAccessGuard guard;
    if (!g_enabled.load(std::memory_order_acquire)) return;
    if (g_enableResetPending.exchange(false, std::memory_order_acq_rel)) {
        // Re-enabling Auto in the same LocalGs must clear pending plans and
        // action gates, but it cannot refund stock passes already spent in
        // this match. A genuinely new LocalGs below still resets the count.
        const int roundRecycles = g_actualRecycleCount;
        resetRoundState(false);
        g_actualRecycleCount = roundRecycles;
    }
    g_accessStage.store(2, std::memory_order_release);
    if (!g_ui && g_getUi) {
        // capture_ui also uses the same guard, so capture directly while the
        // tick owns access rather than re-entering through its public path.
        void* ui = g_getUi(nullptr);
        if (ui && g_ui != ui) {
            void* newGs = *(void**)((uint8_t*)ui + OFF_SOL_UI_LOCAL_GS);
            const int64_t newStart = newGs
                ? *(const int64_t*)((const uint8_t*)newGs + OFF_SOL_START_TIME) : 0;
            const bool sameRound = newGs && newGs == g_lastLocalGs &&
                (!newStart || !g_lastRoundStartTime ||
                 newStart == g_lastRoundStartTime);
            const int roundRecycles = g_actualRecycleCount;
            g_ui = ui;
            if (!sameRound) g_lastLocalGs = nullptr;
            resetRoundState(true, !sameRound);
            if (sameRound) g_actualRecycleCount = roundRecycles;
            SOL_LOG("captured UI=%p", g_ui);
        }
    }
    const uint64_t now = nowMs();
    if (!solitaire_is_ready()) {
        if (now - g_lastDiagMs > 2500) {
            g_lastDiagMs = now;
            SOL_LOG("waiting ready ui=%p str=%p stack=%p rayMove=%p talonUi=%p", g_ui,
                    g_stringNew, g_getStackCards, g_rayMove, g_talonUi);
        }
        return;
    }
    void* localGs = *(void**)((uint8_t*)g_ui + OFF_SOL_UI_LOCAL_GS);
    if (!localGs) {
        if (now - g_lastDiagMs > 2500) {
            g_lastDiagMs = now;
            SOL_LOG("waiting LocalGs ui=%p", g_ui);
        }
        return;
    }
    boostFrameRateIfNeeded(now);
    const int64_t startTime = *(const int64_t*)((const uint8_t*)localGs + OFF_SOL_START_TIME);
    if (localGs != g_lastLocalGs ||
        (startTime && g_lastRoundStartTime && startTime != g_lastRoundStartTime)) {
        g_lastLocalGs = localGs;
        g_lastRoundStartTime = startTime;
        resetRoundState(true, true);
        SOL_LOG("new solitaire round gs=%p startTime=%lld", localGs,
                (long long)startTime);
    } else if (!g_lastRoundStartTime && startTime) {
        g_lastRoundStartTime = startTime;
    }
    // A genuine round transition owns a full-capacity reset. Wait only until
    // the cancelled route-tail slice has destroyed its local frontier, then
    // purge allocator pages before the new deal starts planning.
    if (!finishRoundMemoryReleaseIfReady(now)) return;
    const int configuredDrawCount = *(int*)((uint8_t*)localGs + OFF_SOL_CARDS_PER_TALON_FLIP);
    if (configuredDrawCount >= 1 && configuredDrawCount <= 3 &&
        configuredDrawCount != g_cardsPerTalonFlip) {
        g_cardsPerTalonFlip = configuredDrawCount;
        clearPlannerFrontier();
        SOL_LOG("using LocalGs cards-per-flip=%d", g_cardsPerTalonFlip);
    }
    logNativeStackKeys(localGs);
    discoverStackPrefix(localGs);
    logHiddenStateSnapshot(localGs);
    logFullSnapshot(localGs);

    // Talon flips settle sooner than card drags. This only shortens visual wait;
    // chooseAndExecuteExpertMove still validates g_pendingExpectedKey and will
    // not release a second action until the physical board is synchronized.
    const bool humanPacing = humanSpeedMode();
    const uint64_t settleMs = g_lastActionWasTalon
        ? (humanPacing ? SOL_HUMAN_TALON_ACTION_MS
                       : SOL_TALON_ACTION_SETTLE_MS) + g_pacingBackoffMs
        : (humanPacing ? SOL_HUMAN_CARD_ACTION_MS
                       : SOL_CARD_ACTION_SETTLE_MS) + g_pacingBackoffMs;
    if (g_lastMoveMs != 0 && now - g_lastMoveMs < settleMs) return;
    // Use the game's own UI synchronization flag instead of a conservative
    // fixed sleep. This releases the next search/action on the first safe
    // frame, while a bounded watchdog prevents a stale flag from freezing a
    // round forever.
    const bool uiUpdating =
        *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_IS_UPDATING_UI) != 0;
    // With a flight outstanding, keep observing through the animation instead
    // of returning here. pendingFlightBlocksObservation() applies the identical
    // uiUpdating/watchdog gate before it settles, so nothing is dispatched any
    // earlier; but its first-observation + stable window is already satisfied
    // when the gate opens, saving one whole frame per action. (IDA: DropCard
    // applies OnMoveCard to LocalGs synchronously, so the post-move board is
    // already final while Unity animates it.)
    if (uiUpdating && g_lastMoveMs != 0 && !g_waitingForMoveResult &&
        now - g_lastMoveMs < effectiveUiWatchdogMs()) return;

    const SolitaireStackLists stackLists = currentStackLists(localGs);
    std::array<std::vector<Card>, 7> tableau;
    for (int i = 0; i < 7; ++i) tableau[i] = stackCardsFromList(stackLists.tableau[i]);
    std::vector<Card> waste = stackCardsFromList(stackLists.waste);
    std::vector<Card> talon = stackCardsFromList(stackLists.talon);
    std::array<std::vector<Card>, 4> foundations;
    for (int i = 0; i < 4; ++i) foundations[i] = stackCardsFromList(stackLists.foundations[i]);
    int totalCards = (int)talon.size() + (int)waste.size();
    for (const auto& pile : tableau) totalCards += (int)pile.size();
    for (const auto& pile : foundations) totalCards += (int)pile.size();
    // Undo the opponent reveal leak before anything reads visibility. Only a
    // complete snapshot may advance the per-column history.
    normalizeTableauVisibility(tableau, totalCards == 52);
    hydratePerfectInformation(localGs, tableau, talon, waste, foundations);
    // The UI object exists briefly before the deal populates all stacks. Do
    // not poison the persistent planner/dead cache with that empty snapshot.
    if (totalCards != 52) {
        if (now - g_lastDiagMs > 1000) {
            g_lastDiagMs = now;
            SOL_LOG("[INCOMPLETE_STATE] total=%d talon=%d waste=%d ui=%d localSync=%d",
                    totalCards, (int)talon.size(), (int)waste.size(),
                    uiUpdating ? 1 : 0,
                    *(const uint8_t*)((const uint8_t*)localGs + OFF_SOL_LOCAL_SYNC_APPLIED) ? 1 : 0);
        }
        return;
    }
    const GameState expertState = makeGameState(tableau, foundations, talon, waste);
    g_accessStage.store(3, std::memory_order_release);
    // Reconcile/settle an outstanding action before this snapshot changes
    // progress epochs, loop history, checkpoints, or emergency state.
    if (pendingFlightBlocksObservation(localGs, expertState, now)) {
        // Server mode: keep up to SOL_SERVER_PIPELINE_MAX plan steps in flight.
        g_dispatchDeferredThisTick = false;
        const std::string observedPhysicalKey = physicalPositionKey(expertState);
        while (extendServerPipeline(localGs, now, observedPhysicalKey)) {}
        return;
    }
    rememberTableauPosition(tableau, now);
    logPerfectPlannerSnapshot(expertState);
    // A completed board briefly remains attached to the UI before the result
    // screen replaces LocalGs.  It has no legal moves by definition and must
    // not be reported/recovered as a strategic stall.
    if (virtualFoundationCount(expertState) == 52) return;
    // Observe the fully hydrated state before adding it to the generic recent
    // history. Strategic progress owns a separate epoch/visit model so normal
    // Draw-3 traversal is not mistaken for a repeated dead position.
    observeStrategicProgress(expertState, now);
    rememberSolverState(expertState, now);
    if (now - g_lastDiagMs > 2500) {
        g_lastDiagMs = now;
        SOL_LOG("state gs=%p tableau=[%d,%d,%d,%d,%d,%d,%d] waste=%d", localGs,
                (int)tableau[0].size(), (int)tableau[1].size(), (int)tableau[2].size(),
                (int)tableau[3].size(), (int)tableau[4].size(), (int)tableau[5].size(),
                (int)tableau[6].size(), (int)waste.size());
    }

    // KING routes a Talon-only root to a proven useful Waste exit inside the
    // unified chooser.  Do not fire the old blind one-click shortcut here: it
    // could consume a fourth paid pass before comparing the complete orbit.

    // One unified planner ranks foundation, tableau and stock actions together.
    // This is what allows it to skip a tempting legal move when a calculated
    // Talon branch unlocks a materially stronger continuation.
    g_dispatchDeferredThisTick = false;
    g_accessStage.store(4, std::memory_order_release);
    if (chooseAndExecuteExpertMove(localGs, expertState)) {
        g_lastMoveMs = now;
        // Server mode: send the following known plan steps without waiting.
        const std::string observedPhysicalKey = physicalPositionKey(expertState);
        while (extendServerPipeline(localGs, now, observedPhysicalKey)) {}
        return;
    }
    g_accessStage.store(5, std::memory_order_release);
    SOL_TRACE("[SEARCH] choice-returned no-dispatch frontier=%d roots=%d",
            (int)g_persistentPlanner.frontier.size(), (int)g_persistentPlanner.roots.size());
    // A verified action promoted a queued step: it is still in flight, so
    // top the queue up instead of running the idle/emergency paths below.
    if (g_waitingForMoveResult) {
        const std::string observedPhysicalKey = physicalPositionKey(expertState);
        while (extendServerPipeline(localGs, now, observedPhysicalKey)) {}
        return;
    }

    // A missing view/controller is a temporary UI condition, not a reason to
    // run emergency planning or memoize the board as having no legal move.
    if (g_dispatchDeferredThisTick) return;

    // Do not wait for a large but non-productive frontier forever. A bounded
    // stock-orbit proof may establish that Talon exposes a genuinely new exit;
    // dispatch that one reversible edge automatically so manual wake-up taps
    // are never required.
    if (!g_waitingForMoveResult &&
        attemptProvenTalonLiveness(localGs, expertState, now)) {
        clearPlannerFrontier();
        g_lastMoveMs = now;
        return;
    }

    // The search normally returns within one or two frames.  If its frontier
    // is exhausted, or three complete stock passes produced no progress, use
    // a narrow legal-progress escape hatch rather than leaving the bot idle.
    if (!g_waitingForMoveResult && g_persistentPlanner.frontier.empty() &&
        attemptEmergencyProgress(localGs, expertState)) {
        clearPlannerFrontier();
        g_lastMoveMs = now;
        return;
    }
    logStalledExpertState(expertState, now);
    g_accessStage.store(6, std::memory_order_release);
    SOL_TRACE("[TICK_EXIT] after-stall");
}

void solitaire_tick() {
    try {
        solitaire_tick_impl();
    } catch (...) {
        const int owner = g_accessOwnerTid.load(std::memory_order_acquire);
        SOL_LOG("[TICK_EXCEPTION] tid=%d owner=%d stage=%d",
                gettid(), owner, g_accessStage.load(std::memory_order_acquire));
        // The access guard normally unwinds before this catch. If a foreign
        // native throw skipped it, release only our own Unity thread's gate.
        if (owner == gettid()) {
            g_accessStage.store(0, std::memory_order_release);
            g_accessAcquiredMs.store(0, std::memory_order_release);
            g_accessOwnerTid.store(0, std::memory_order_release);
            g_solitaireAccess.clear(std::memory_order_release);
        }
        g_enableResetPending.store(true, std::memory_order_release);
        g_lastMoveMs = nowMs();
        g_pacingBackoffMs = SOL_MAX_PACING_BACKOFF_MS;
    }
}
