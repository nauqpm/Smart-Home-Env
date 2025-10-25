# HumanBehavior (mã thay thế / nâng cấp)
import numpy as np
import random

class HumanBehavior:
    """
    Mô phỏng hành vi con người (gia đình 4 người) theo kế hoạch:
    - từng thành viên có lịch cá nhân
    - day_type: weekday / weekend / holiday (nghỉ đột xuất)
    - occupancy: giá trị 0.0 - 1.0 tương ứng tỷ lệ người ở nhà
    - appliance_usage: dict thiết bị -> mảng xác suất hoặc boolean (T,)
    """

    def __init__(self, num_people=4, T=24, random_day_off_prob=0.08, seed=None, weather=None):
        """
        :param num_people: số người trong nhà (mặc định 4)
        :param T: timesteps / ngày (mặc định 24 giờ)
        :param random_day_off_prob: xác suất 1 ngày là 'holiday' (nghỉ đột xuất)
        :param seed: seed cho reproducibility
        :param weather: optional string 'sunny','mild','cloudy','rainy','stormy' (nếu None thì tự sinh)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_people = num_people
        self.T = T
        self.random_day_off_prob = random_day_off_prob
        self.weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # members: list of dict (role, schedule)
        # schedule: list of tuples (start_hour, end_hour) where member is OUT (not at home).
        # here: bố/mẹ đi làm 9-18, con lớn 7-17, con nhỏ 7-17
        self.members = [
            {"name":"father", "type":"adult", "out_periods":[(9,18)]},
            {"name":"mother", "type":"adult", "out_periods":[(9,18)]},
            {"name":"child1", "type":"child", "out_periods":[(7,17)]},
            {"name":"child2", "type":"child", "out_periods":[(7,17)]},
        ]
        # nếu num_people khác 4, cắt/gia tăng từ trên
        if num_people != 4:
            self._adjust_members(num_people)

        # day type: weekday / weekend / holiday
        self.day_type = self._get_day_type()

        # weather
        self.weather = weather if weather in ["sunny","mild","cloudy","rainy","stormy"] else self._sample_weather()

        # tạo occupancy và appliance usage
        self.occupancy = self._generate_occupancy()
        self.device_usage = self._generate_device_usage()

    def _adjust_members(self, n):
        # simple adjust: truncate or extend by copying child profiles
        if n < 4:
            self.members = self.members[:n]
        else:
            while len(self.members) < n:
                self.members.append({"name":f"extra{len(self.members)}", "type":"adult", "out_periods":[(9,18)]})

    def _get_day_type(self):
        r = random.random()
        if r < self.random_day_off_prob:
            return "holiday"
        day_index = random.randint(0,6)
        return "weekend" if day_index >= 5 else "weekday"

    def _sample_weather(self):
        # simple prior
        return random.choice(["sunny","mild","cloudy","rainy","stormy"])

    def _is_member_home(self, member, hour):
        """
        Trả về True nếu member ở nhà tại giờ 'hour'.
        - nếu day_type == weekend/holiday: ở nhà nhiều hơn
        - nếu random event: small prob thay đổi (eg. về sớm, về muộn)
        """
        # weekend/holiday: assume at home except sleeping/outings
        if self.day_type in ["weekend","holiday"]:
            # small chance to go out midday
            if random.random() < 0.07:
                return False
            # otherwise mostly home; but still respect sleeping hours
            if 0 <= hour < 6:
                return False  # asleep (but considered at home; for occupancy we treat asleep as 'at home' but lower activity)
            return True

        # weekday: use out_periods to determine OUT; also allow small randomness
        for (s,e) in member.get("out_periods", []):
            if s <= hour < e:
                # small chance to be at home (WFH / sick)
                if random.random() < 0.08:
                    return True
                return False
        # not in out_periods -> at home
        # early morning (0-6): often sleeping -> still at home (counted)
        # late night: at home
        # small chance of being out
        if random.random() < 0.03:
            return False
        return True

    def _generate_occupancy(self):
        """
        Trả về mảng length T có giá trị 0..1; dựa trên số người đang ở nhà + noise.
        Giá trị = (n_people_home / total_people) * scaling + small_noise
        """
        occ = np.zeros(self.T)
        for t in range(self.T):
            n_home = 0
            for m in self.members:
                if self._is_member_home(m, t):
                    n_home += 1
            base = n_home / max(1, len(self.members))  # 0..1

            # adjust baseline by time-of-day patterns: e.g., when everyone at evening, boost
            if self.day_type in ["weekend","holiday"]:
                # weekend -> more at home
                base = np.clip(base + 0.05, 0, 1)
            # noise to reflect "partial presence" (e.g., someone pops out)
            noise = np.random.normal(0.0, 0.03)
            occ[t] = float(np.clip(base + noise, 0.0, 1.0))
        return occ

    def _generate_device_usage(self):
        """
        Trả về dict thiết bị -> numpy array (T,) chứa xác suất bật (0..1).
        Sau đó khi cần boolean: sample random < p để quyết định bật hay tắt.
        Devices implemented:
          - lights, fridge, TV, AC, heater, washing_machine, dishwasher, laptop, ev_charger
        """
        devices = {
            "lights": np.zeros(self.T),
            "fridge": np.zeros(self.T),
            "tv_prob": np.zeros(self.T),
            "ac_prob": np.zeros(self.T),
            "heater_prob": np.zeros(self.T),
            "washing_machine_prob": np.zeros(self.T),
            "dishwasher_prob": np.zeros(self.T),
            "laptop_prob": np.zeros(self.T),
            "ev_charger_prob": np.zeros(self.T),
        }

        for t in range(self.T):
            occ = self.occupancy[t]

            # Fridge: always on base consumption -> probability ~1
            devices["fridge"][t] = 0.98

            # Lights: bật nếu có người và tối (giờ >=18 or <6) hoặc trời mưa
            is_dark = (t < 6 or t >= 18)
            if occ > 0.1 and (is_dark or self.weather in ["rainy","stormy"]):
                # xác suất cao hơn khi có nhiều người
                devices["lights"][t] = np.clip(0.5 + 0.5 * occ + np.random.normal(0,0.05), 0, 1)
            else:
                devices["lights"][t] = np.clip(0.05 * occ, 0, 1)

            # TV: buổi tối, cuối tuần: xác suất cao nếu occ>0.4
            if occ > 0.4 and (self.day_type != "weekday" or 18 <= t < 22):
                devices["tv_prob"][t] = np.clip(0.4 + 0.6 * occ, 0, 1)
            else:
                devices["tv_prob"][t] = 0.05 * occ

            # AC: tăng khi trời nóng (weather proxy) or sunny afternoon (12-18)
            # map weather to temp factor (sunny->hot)
            weather_temp_factor = {"sunny":1.0, "mild":0.7, "cloudy":0.5, "rainy":0.3, "stormy":0.2}
            wfactor = weather_temp_factor.get(self.weather, 0.7)
            if occ > 0.3 and (12 <= t < 22):
                devices["ac_prob"][t] = np.clip(0.3 + 0.7 * occ * wfactor, 0, 1)
            else:
                devices["ac_prob"][t] = np.clip(0.05 * occ, 0, 1)

            # Heater: sáng sớm và tối, hoặc cold weather
            if (6 <= t < 7 or 20 <= t < 22) or self.weather in ["rainy","stormy"]:
                devices["heater_prob"][t] = np.clip(0.4 + 0.6 * occ, 0, 1)
            else:
                devices["heater_prob"][t] = 0.02 * occ

            # Washing machine: chủ yếu cuối tuần hoặc tối (19-22) theo xác suất
            if self.day_type != "weekday":
                devices["washing_machine_prob"][t] = 0.25  # nhiều khả năng chạy
            elif 19 <= t < 22:
                devices["washing_machine_prob"][t] = 0.12
            else:
                devices["washing_machine_prob"][t] = 0.02

            # Dishwasher: sau bữa tối nếu có người
            if occ > 0.5 and 20 <= t < 22:
                devices["dishwasher_prob"][t] = 0.45
            else:
                devices["dishwasher_prob"][t] = 0.05 * occ

            # Laptop: học/làm việc (buổi tối và cuối tuần)
            if occ > 0.4 and (self.day_type != "weekday" or 18 <= t < 23):
                devices["laptop_prob"][t] = 0.4 + 0.5 * occ
            elif 7 <= t < 17 and any(m["type"] in ["adult","student"] for m in self.members):
                # low chance daytime for remote work
                devices["laptop_prob"][t] = 0.15 * occ
            else:
                devices["laptop_prob"][t] = 0.02 * occ

            # EV charger: nếu nhà có xe (giả sử có), chủ yếu 19-23
            if 19 <= t < 23 and self.day_type != "weekday":
                devices["ev_charger_prob"][t] = 0.6
            elif 19 <= t < 23:
                devices["ev_charger_prob"][t] = 0.5
            else:
                devices["ev_charger_prob"][t] = 0.02

            # add small randomness to each prob to avoid deterministic patterns
            for k in devices:
                devices[k][t] = float(np.clip(devices[k][t] + np.random.normal(0, 0.03), 0.0, 1.0))

        return devices

    def sample_appliance_states(self):
        """
        Trả về dict thiết bị -> boolean array (T,) bằng cách sample theo xác suất.
        Useful if you want actual ON/OFF traces for a day.
        """
        states = {}
        for k, arr in self.device_usage.items():
            states[k] = np.random.rand(self.T) < arr
        return states

    def generate_daily_behavior(self):
        """
        API tương tự: trả occupancy_profile, appliance_usage_probs
        """
        return self.occupancy, self.device_usage

if __name__ == "__main__":
    # quick demo
    hb = HumanBehavior(seed=42)
    occ, dev = hb.generate_daily_behavior()
    print("day_type:", hb.day_type, "weather:", hb.weather)
    print("occupancy:", np.round(occ,3))
    print("lights probs:", np.round(dev["lights"],3))
