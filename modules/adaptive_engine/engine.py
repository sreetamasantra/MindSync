import time

# How long a state must persist before triggering action (seconds)
STATE_PERSISTENCE = 3

# Cooldown between repeated recommendations (seconds)
ACTION_COOLDOWN = 10

# Action map — what to do for each state
ACTIONS = {
    "Focused":    {
        "message":  "Great focus! Increasing content difficulty.",
        "action":   "increase_difficulty"
    },
    "Distracted": {
        "message":  "You seem distracted. Refocus on the screen!",
        "action":   "attention_alert"
    },
    "Fatigued":   {
        "message":  "You look tired. Take a 5-minute break.",
        "action":   "suggest_break"
    },
    "Confused":   {
        "message":  "Seems like you're confused. Switching to simpler explanation.",
        "action":   "simplify_content"
    }
}

class AdaptiveEngine:
    def __init__(self):
        self.current_state      = None
        self.state_start_time   = None
        self.last_action_time   = {}   # state -> last time action was triggered
        self.action_log         = []   # list of logged actions
        self.last_recommendation = None

    def update(self, state):
        """
        Call this every frame with the current detected state.
        Returns a recommendation dict or None.
        """
        now = time.time()

        # If state changed, reset timer
        if state != self.current_state:
            self.current_state    = state
            self.state_start_time = now
            return None

        # Check if state has persisted long enough
        state_duration = now - self.state_start_time
        if state_duration < STATE_PERSISTENCE:
            return None

        # Check cooldown — don't repeat same action too soon
        last_time = self.last_action_time.get(state, 0)
        if now - last_time < ACTION_COOLDOWN:
            return None

        # Trigger action
        action_data = ACTIONS[state]
        recommendation = {
            "state":     state,
            "message":   action_data["message"],
            "action":    action_data["action"],
            "timestamp": now
        }

        # Update tracking
        self.last_action_time[state] = now
        self.last_recommendation     = recommendation
        self.action_log.append(recommendation)

        return recommendation

    def get_log(self):
        return self.action_log

    def get_last_recommendation(self):
        return self.last_recommendation