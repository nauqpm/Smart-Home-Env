# Tên file: human_behavior.py
# ĐÃ SỬA LỖI: Di chuyển 'generate_month_behavior_with_schedule' vào đúng vị trí.

"""
HumanBehavior - upgraded full version
Features:
- Member profiles with heterogeneity (office_worker, remote_worker, shift_worker, student)
- Proper handling of night (sleep = present but low activity)
- Returns presence_counts, occupancy_ratio, activity_level
- Latent daily activity states to create device correlations (sleep, cooking, watching_tv, working, away)
- Device probability traces and sampled device ON/OFF traces with duration/state machines
- Seasonality-aware weather sampling
- Seedable for reproducibility
Usage:
    hb = HumanBehavior(num_people=4, T=24, seed=42, month=7)
    out = hb.generate_daily_behavior(sample_device_states=True)
    # out contains presence_counts, occupancy_ratio, activity_level, device_probs, device_states
"""
import numpy as np
import random
from collections import defaultdict

class HumanBehavior:
    def __init__(self,
                 num_people=4,
                 T=24,
                 random_day_off_prob=0.08,
                 seed=None,
                 weather=None,
                 month=None,   # 1..12 for seasonality
                 profiles=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_people = max(1, int(num_people))
        self.T = int(T)
        self.random_day_off_prob = float(random_day_off_prob)
        self.month = month  # if provided, used for seasonal bias
        self.weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # default profiles pool
        self.profile_library = {
            "office_worker": {"out_periods":[(9,18)], "prob_wfh":0.08, "late_return_prob":0.05, "sleep_offset":0},
            "remote_worker": {"out_periods":[], "prob_wfh":0.75, "late_return_prob":0.02, "sleep_offset":0},
            "shift_worker": {"out_periods":[(22,6)], "prob_wfh":0.05, "late_return_prob":0.15, "sleep_offset":0},
            "student": {"out_periods":[(7,17)], "prob_wfh":0.15, "late_return_prob":0.08, "sleep_offset":0}
        }

        # create members (assign heterogeneous profiles)
        self.members = []
        if profiles and len(profiles) >= self.num_people:
            # user provided explicit profile list (strings)
            for i in range(self.num_people):
                pname = profiles[i]
                p = self.profile_library.get(pname, self.profile_library["office_worker"])
                self.members.append({"name": f"m{i}", "profile": pname, "out_periods": p["out_periods"], "meta": p})
        else:
            # default mix: father/mother/student/student style but randomized if >4
            base = [
                ("father", "office_worker"),
                ("mother", "office_worker"),
                ("child1", "student"),
                ("child2", "student")
            ]
            for i in range(self.num_people):
                if i < len(base):
                    name, prof = base[i]
                else:
                    # assign random profile for extra members
                    prof = random.choice(list(self.profile_library.keys()))
                    name = f"extra{i}"
                p = self.profile_library.get(prof, self.profile_library["office_worker"])
                self.members.append({"name": name, "profile": prof, "out_periods": p["out_periods"], "meta": p})

        # day type & weather
        self.day_type = self._get_day_type()
        self.weather = weather if weather in ["sunny","mild","cloudy","rainy","stormy"] else self._sample_weather()

        # Prepare per-hour containers
        self.presence_counts = np.zeros(self.T, dtype=int)
        self.occupancy_ratio = np.zeros(self.T, dtype=float)
        self.activity_level = np.zeros(self.T, dtype=float)

        # device list and default power can be used externally
        self.devices = ["lights","fridge","tv","ac","heater","washing_machine","dishwasher","laptop","ev_charger"]

        # device probability traces (per-hour)
        self.device_probs = {d: np.zeros(self.T, dtype=float) for d in self.devices}
        # optional sampled device on/off traces (bool) after applying duration rules
        self.device_states = {d: np.zeros(self.T, dtype=bool) for d in self.devices}

        # internal state machines for duration-based devices
        self._state = {
            "washing": {"running":False, "remaining":0.0},
            "dishwasher": {"running":False, "remaining":0.0},
            "ev": {"running":False, "remaining":0.0}
        }

        # latent activity mapping: each state -> per-device base prob (0..1)
        self.latent_states = ["sleep", "away", "home_idle", "cooking", "watching_tv", "working"]
        self.latent_map = self._build_latent_map()

        # finally generate base behavior
        self._generate_presence_and_activity()
        self._generate_device_probabilities()
        # device_states are not sampled by default; call generate_daily_behavior(sample_device_states=True) to get ON/OFF traces

    def _get_day_type(self):
        r = random.random()
        if r < self.random_day_off_prob:
            return "holiday"
        day_index = random.randint(0,6)
        return "weekend" if day_index >= 5 else "weekday"

    def _seasonal_weather_bias(self):
        # crude seasonal bias: summer (6-8) -> more sunny, winter months -> more cloudy/rainy
        if self.month is None:
            return None
        m = int(self.month)
        if m in (6,7,8):
            return "summer"
        if m in (12,1,2):
            return "winter"
        return "neutral"

    def _sample_weather(self):
        # use rough seasonal bias if available
        season = self._seasonal_weather_bias()
        base = ["sunny","mild","cloudy","rainy","stormy"]
        weights = [0.2,0.25,0.25,0.2,0.1]
        if season == "summer":
            weights = [0.4,0.3,0.15,0.1,0.05]
        elif season == "winter":
            weights = [0.05,0.25,0.35,0.25,0.1]
        return random.choices(base, weights=weights, k=1)[0]

    def _build_latent_map(self):
        # baseline conditional probabilities for each device given latent activity
        # these are tunable constants; they define correlations
        m = {}
        m["sleep"] = {"lights":0.02, "tv":0.01, "ac":0.25, "heater":0.3, "washing_machine":0.0, "dishwasher":0.0, "laptop":0.02, "ev_charger":0.02, "fridge":0.98}
        m["away"] = {"lights":0.01, "tv":0.01, "ac":0.05, "heater":0.05, "washing_machine":0.02, "dishwasher":0.02, "laptop":0.01, "ev_charger":0.02, "fridge":0.98}
        m["home_idle"] = {"lights":0.3, "tv":0.15, "ac":0.4, "heater":0.3, "washing_machine":0.02, "dishwasher":0.02, "laptop":0.2, "ev_charger":0.05, "fridge":0.98}
        m["cooking"] = {"lights":0.85, "tv":0.05, "ac":0.25, "heater":0.15, "washing_machine":0.1, "dishwasher":0.3, "laptop":0.05, "ev_charger":0.02, "fridge":0.99}
        m["watching_tv"] = {"lights":0.75, "tv":0.95, "ac":0.45, "heater":0.25, "washing_machine":0.01, "dishwasher":0.01, "laptop":0.05, "ev_charger":0.02, "fridge":0.98}
        m["working"] = {"lights":0.35, "tv":0.02, "ac":0.5, "heater":0.2, "washing_machine":0.01, "dishwasher":0.01, "laptop":0.8, "ev_charger":0.02, "fridge":0.98}
        return m

    def _is_member_home(self, member, hour):
        """
        Returns True if member is physically present at home at 'hour'.
        Sleep hours -> still considered present.
        For shift_worker with out_period crossing midnight, handle wrap-around.
        """
        # weekend/holiday: more likely at home, but can go out briefly
        if self.day_type in ["weekend", "holiday"]:
            if random.random() < 0.07:
                return False
            return True

        # weekday: respect out_periods with small WFH/sick chance
        for (s,e) in member.get("out_periods", []):
            # handle wrap-around e < s (overnight shift)
            if s <= e:
                in_out = (s <= hour < e)
            else:
                in_out = (hour >= s or hour < e)
            if in_out:
                # small chance to be at home (WFH or unexpected)
                if random.random() < 0.08:
                    return True
                return False
        # otherwise at home with small chance to step out
        if random.random() < 0.03:
            return False
        return True

    def _generate_presence_and_activity(self):
        """
        Populate presence_counts, occupancy_ratio, activity_level.
        activity_level in [0,1] corresponds to intensity of in-home activity (affects device usage likelihood).
        """
        for t in range(self.T):
            n_home = 0
            for m in self.members:
                if self._is_member_home(m, t):
                    n_home += 1
            self.presence_counts[t] = n_home
            occ = n_home / max(1, len(self.members))
            # baseline activity by hour (sleep low, evening high)
            if 0 <= t < 6:
                base_act = 0.10
            elif 6 <= t < 9:
                base_act = 0.35
            elif 9 <= t < 17:
                base_act = 0.25
            elif 17 <= t < 21:
                base_act = 0.75
            else:
                base_act = 0.35
            # adjust for day type
            if self.day_type in ["weekend", "holiday"]:
                base_act = min(1.0, base_act + 0.1)
            # scale by number of people and add noise
            act = base_act * (n_home / max(1, len(self.members)))
            act += np.random.normal(0.0, 0.05)
            self.activity_level[t] = float(np.clip(act, 0.0, 1.0))
            self.occupancy_ratio[t] = float(np.clip(occ + np.random.normal(0.0, 0.02), 0.0, 1.0))

    def _choose_latent_state(self, t):
        """
        Heuristic to choose latent activity state at hour t using activity_level and occupancy.
        Returns one of self.latent_states.
        """
        occ = self.presence_counts[t]
        act = self.activity_level[t]
        # sleeping window preference
        if 0 <= t < 6:
            return "sleep"
        # away if no one at home
        if occ == 0:
            return "away"
        # dinner window
        if 17 <= t < 20 and act > 0.4:
            return "cooking"
        # watching TV typical 19-23 if activity high
        if 19 <= t < 23 and act > 0.4:
            # randomly pick watching_tv vs home_idle
            return "watching_tv" if random.random() < 0.8 else "home_idle"
        # working if someone at home and high laptop probability (approx from profiles)
        # a quick heuristic: if any member is remote_worker -> working during day
        if 9 <= t < 17:
            if any(m["profile"] == "remote_worker" for m in self.members) and act > 0.1:
                return "working"
        # fallback
        return "home_idle"

    def _generate_device_probabilities(self):
        """
        Build device_probs per hour by combining latent_map, weather, and occupancy/activity.
        """
        # weather temp factor affects AC/heater
        weather_temp_factor = {"sunny":1.0, "mild":0.8, "cloudy":0.6, "rainy":0.4, "stormy":0.25}
        wfactor = weather_temp_factor.get(self.weather, 0.7)

        for t in range(self.T):
            latent = self._choose_latent_state(t)
            base = self.latent_map.get(latent, {})
            occ_ratio = self.occupancy_ratio[t]
            act = self.activity_level[t]

            # combine: base prob scaled by occupancy/activity plus small noise
            for d in self.devices:
                p_base = base.get(d, 0.01)
                # fridge baseline near 1
                if d == "fridge":
                    p = max(0.95, p_base)
                else:
                    # scale logic:
                    # lights: strongly affected by darkness (hour)
                    if d == "lights":
                        is_dark = (t < 6 or t >= 18) or (self.weather in ["rainy","stormy"] and t >= 6)
                        p = p_base * (0.5 + 0.5 * occ_ratio) * (1.3 if is_dark else 1.0)
                    elif d == "ac":
                        p = p_base * (0.4 + 0.6 * occ_ratio) * wfactor
                    elif d == "heater":
                        # heater more when cold (rainy/stormy) or morning/evening
                        cold = (self.weather in ["rainy","stormy"])
                        p = p_base * (0.4 + 0.6 * occ_ratio) * (1.2 if cold else 1.0)
                    elif d in ["washing_machine", "dishwasher"]:
                        # higher on weekends/evenings
                        if self.day_type in ["weekend","holiday"]:
                            p = p_base * 1.8 * occ_ratio
                        elif 18 <= t < 22:
                            p = p_base * 1.2 * occ_ratio
                        else:
                            p = p_base * 0.5 * occ_ratio
                    elif d == "tv":
                        p = p_base * (0.2 + 0.8 * act)
                    elif d == "laptop":
                        p = p_base * (0.1 + 0.9 * (1.0 if any(m["profile"]=="remote_worker" for m in self.members) else occ_ratio))
                    elif d == "ev_charger":
                        # evening preference for charging
                        if 19 <= t < 24:
                            p = 0.5 * occ_ratio
                        else:
                            p = 0.02
                    else:
                        p = p_base * (0.3 + 0.7 * occ_ratio)

                # add gaussian noise and clip
                p = p + np.random.normal(0.0, 0.03)
                p = float(np.clip(p, 0.0, 1.0))
                self.device_probs[d][t] = p

    # Duration/state machine helpers for devices that run multi-hour
    def _maybe_start_washing(self, t):
        st = self._state["washing"]
        if not st["running"]:
            start_prob = self.device_probs["washing_machine"][t]
            if random.random() < start_prob:
                st["running"] = True
                # sample duration in hours
                st["remaining"] = random.choices([1.0, 1.5, 2.0], weights=[0.6,0.25,0.15], k=1)[0]
        else:
            st["remaining"] -= 1.0
            if st["remaining"] <= 0:
                st["running"] = False

    def _maybe_start_dishwasher(self, t):
        st = self._state["dishwasher"]
        if not st["running"]:
            start_prob = self.device_probs["dishwasher"][t]
            if random.random() < start_prob:
                st["running"] = True
                st["remaining"] = random.choices([1.0, 1.5], weights=[0.8,0.2], k=1)[0]
        else:
            st["remaining"] -= 1.0
            if st["remaining"] <= 0:
                st["running"] = False

    def _maybe_start_ev(self, t):
        st = self._state["ev"]
        if not st["running"]:
            start_prob = self.device_probs["ev_charger"][t]
            if random.random() < start_prob:
                st["running"] = True
                # sample number of hours to charge (typical 2-6 hours)
                st["remaining"] = random.randint(2,6)
        else:
            st["remaining"] -= 1.0
            if st["remaining"] <= 0:
                st["running"] = False

    def sample_device_states(self):
        """
        From device_probs and internal duration models, produce device_states bool array per hour.
        This maintains durations for washing, dishwasher, ev charger.
        """
        # reset state machines
        self._state["washing"] = {"running":False, "remaining":0.0}
        self._state["dishwasher"] = {"running":False, "remaining":0.0}
        self._state["ev"] = {"running":False, "remaining":0.0}
        # clear previous device states
        for d in self.devices:
            self.device_states[d][:] = False

        for t in range(self.T):
            # latent-based sampling for correlated devices: sample latent then sample devices conditioned
            latent = self._choose_latent_state(t)
            # for each device except duration-aware ones, sample Bernoulli with device_probs
            for d in self.devices:
                if d == "washing_machine":
                    # use washing state machine
                    self._maybe_start_washing(t)
                    self.device_states["washing_machine"][t] = self._state["washing"]["running"]
                elif d == "dishwasher":
                    self._maybe_start_dishwasher(t)
                    self.device_states["dishwasher"][t] = self._state["dishwasher"]["running"]
                elif d == "ev_charger":
                    self._maybe_start_ev(t)
                    self.device_states["ev_charger"][t] = self._state["ev"]["running"]
                else:
                    p = self.device_probs[d][t]
                    # conditional tweak: if latent strongly suggests device, boost p
                    latent_boost = self.latent_map.get(latent, {}).get(d, 0.0)
                    p_cond = np.clip(0.6 * p + 0.4 * latent_boost, 0.0, 1.0)
                    self.device_states[d][t] = (random.random() < p_cond)

            # fridge always on with small duty cycle - but we mark as True for baseline
            self.device_states["fridge"][t] = True

        return {d: self.device_states[d].astype(bool).tolist() for d in self.devices}

    def generate_daily_behavior(self, sample_device_states=False):
        """
        Public API:
        returns dict with:
         - presence_counts (list of int)
         - occupancy_ratio (list of float)
         - activity_level (list of float)
         - weather (string)
         - day_type (string)
         - device_probs (dict device -> list[float])
         - device_states (optional) (dict device -> list[bool]) if sample_device_states=True
        """
        out = {
            "presence_counts": self.presence_counts.tolist(),
            "occupancy_ratio": self.occupancy_ratio.tolist(),
            "activity_level": self.activity_level.tolist(),
            "weather": self.weather,
            "day_type": self.day_type,
            "device_probs": {d: self.device_probs[d].tolist() for d in self.devices}
        }
        if sample_device_states:
            out["device_states"] = self.sample_device_states()
        return out

    # === PHẦN MỞ RỘNG: Event Scheduler cho mô phỏng nhiều ngày ===
    # === ĐÃ DI CHUYỂN VÀO BÊN TRONG LỚP HumanBehavior (SỬA LỖI) ===
    def generate_month_behavior_with_schedule(self, start_day="monday", days=30):
        """
        Tạo dữ liệu hành vi cho nhiều ngày (multi-day simulation) với lịch thực tế.
        Mỗi ngày có thể là: weekday, weekend, holiday, sick_day, guest_day.
        Không ảnh hưởng đến các hàm cũ.
        """
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }
        start_idx = weekday_map.get(start_day.lower(), 0)

        # Xác suất cho các sự kiện đặc biệt
        special_probs = {
            "holiday": 0.05,
            "sick_day": 0.03,
            "guest_day": 0.05,
            "vacation": 0.02
        }

        month_behavior = {}
        # Dùng seed ngẫu nhiên cho mỗi ngày
        day_seed = random.randint(0, 1000000)

        for i in range(days):
            weekday_idx = (start_idx + i) % 7
            # Mặc định loại ngày
            if weekday_idx in [5, 6]:
                day_type = "weekend"
            else:
                day_type = "weekday"

            # Gán ngẫu nhiên sự kiện đặc biệt
            for ev, p in special_probs.items():
                if random.random() < p:
                    day_type = ev
                    break

            # --- SỬA LỖI LOGIC ---
            # Tạo một đối tượng HumanBehavior MỚI cho mỗi ngày
            # thay vì chỉ gọi lại hàm generate_daily_behavior trên cùng 1 đối tượng.
            # Điều này đảm bảo tính ngẫu nhiên và logic của ngày (self.day_type) được áp dụng đúng.

            # Tạo hb mới cho ngày này, dùng seed khác nhau
            hb_day = HumanBehavior(num_people=self.num_people, T=self.T, seed=(day_seed + i), month=self.month)
            # Ghi đè day_type bằng loại sự kiện đã chọn
            hb_day.day_type = day_type

            # Chạy lại các hàm generator với day_type mới này
            # (Các hàm này sẽ sửa đổi đối tượng hb_day)
            hb_day._generate_presence_and_activity()
            hb_day._generate_device_probabilities()

            # Lấy hành vi của ngày đó
            behavior = hb_day.generate_daily_behavior(sample_device_states=True)
            behavior["event_type"] = day_type
            month_behavior[i] = behavior

        return month_behavior

# === KẾT THÚC LỚP HumanBehavior ===


class EventScheduler:
    """
    Tạo lịch multi-day cho 1 tháng với các ngày weekday/weekend/holiday.
    """
    def __init__(self, days=30, seed=None, sick_prob=0.05, holiday_prob=0.1):
        if seed is not None:
            random.seed(seed)
        self.days = days
        self.sick_prob = sick_prob
        self.holiday_prob = holiday_prob
        self.weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def generate_month_plan(self):
        """
        Trả về danh sách day_type ("weekday", "weekend", "holiday") cho từng ngày.
        """
        plan = []
        for d in range(self.days):
            weekday = self.weekdays[d % 7]
            if random.random() < self.holiday_prob:
                day_type = "holiday"
            elif random.random() < self.sick_prob:
                day_type = "holiday"
            elif weekday in ["Sat", "Sun"]:
                day_type = "weekend"
            else:
                day_type = "weekday"
            plan.append(day_type)
        return plan


def simulate_month(num_people=4, month_days=30, seed=42):
    """
    Mô phỏng hành vi trong 1 tháng, trả về list daily_behavior (30 phần tử)
    mỗi phần là output của generate_daily_behavior()
    """
    scheduler = EventScheduler(days=month_days, seed=seed)
    plan = scheduler.generate_month_plan()

    results = []
    for i, dtype in enumerate(plan):
        hb = HumanBehavior(num_people=num_people, T=24, seed=seed+i)
        # override day_type
        hb.day_type = dtype
        day_behavior = hb.generate_daily_behavior(sample_device_states=True)
        results.append(day_behavior)

    return results




if __name__ == "__main__":
    # chạy thử mô phỏng 7 ngày
    month_data = simulate_month(num_people=4, month_days=7, seed=123)
    import json
    print(json.dumps(month_data, indent=2))