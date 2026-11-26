import csv,re
import pandas as pd

#地址字典
us_states = {'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado','CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'}
canada_states = {'AB':'Alberta', 'BC': 'British Columbia', 'MB': 'Manitoba', 'NB': 'New Brunswick', 'NL': 'Newfoundland', 'NT': 'Northwest Territories', 'NS': 'Nova Scotia', 'NU': 'Nunavut', 'ON': 'Ontario', 'PE': 'Prince Edward Island', 'QC': 'Quebec', 'SK': 'Saskatchewan', 'YT': 'Yukon'}
australia_states = ['Australian Capital Territory', 'New South Wales', 'Northern Territory', 'Queensland', 'South Australia', 'Tasmania', 'Victoria', 'Western Australia']

def read_place_lookup(events_path):
    # 1. 准备国家集合和城市查询字典
    countries = set()
    city_to_places = {}

    # 2. 逐行读取官方活动属性
    with open(
        events_path,
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            city = row["city"].strip()
            state = row["state"].strip()
            country = row["country"].strip()

            # 与原位置处理中的国家名称修正保持一致
            if country == "Democratic Republic Congo":
                country = "Democratic Republic of the Congo"

            # 3. 记录已知国家
            if country:
                countries.add(country)

            # 4. 记录城市可能对应的地区
            if city and country:
                if city not in city_to_places:
                    city_to_places[city] = set()

                city_to_places[city].add((state, country))

    print("地名字典建立结束")
    return countries, city_to_places

def clean_place_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    for separator in [",", ";", "|"]:
        text = text.replace(separator, " ")

    return " ".join(text.split())


def contains_place(text, place):
    place = clean_place_text(place)

    if not place:
        return False

    return f" {place} " in f" {text} "


def parse_location(location, countries, city_to_places):
    # 1. 清洗用户位置
    text = clean_place_text(location)

    if not text:
        return []

    # 2. 找用户明确写出的国家
    country_matches = set()

    for country in countries:
        if contains_place(text, country):
            country_matches.add(country)

    state_matches = find_state_matches(location)

    # 如果明确写了国家，只保留属于该国家的州
    if country_matches:
        state_matches = {
            (state, country)
            for state, country in state_matches
            if country in country_matches
        }

    # 未明确写国家时，可以从州信息推断国家
    if not country_matches and state_matches:
        country_matches = {
            country for state, country in state_matches
        }

    # 3. 查找文本中的城市名
    city_matches = []

    for city in city_to_places:
        if contains_place(text, city):
            city_matches.append(city)

    # 4. 优先保留名称更完整的城市
    if city_matches:
        longest_length = max(
            len(clean_place_text(city))
            for city in city_matches
        )

        city_matches = [
            city
            for city in city_matches
            if len(clean_place_text(city)) == longest_length
        ]

    # 5. 保存可能的位置
    results = []

    for city in sorted(city_matches):
        for state, country in sorted(city_to_places[city]):
            if country_matches and country not in country_matches:
                continue

            state = canonical_state(state, country)

            # 已识别出州时，进一步筛选同名城市
            if state_matches:
                compatible = any(
                    clean_place_text(state) == clean_place_text(known_state)
                    and country == known_country
                    for known_state, known_country in state_matches
                )

                if not compatible:
                    continue

            results.append({
                "city": city,
                "state": state,
                "country": country,
            })

    # 6. 没识别出城市，但识别出国家
    if not results:
        if state_matches:
            for state, country in sorted(state_matches):
                results.append({
                    "state": state,
                    "country": country,
                })
        else:
            for country in sorted(country_matches):
                results.append({"country": country})

    return results


def find_state_matches(location):
    # 1. 清洗后的位置文本，用于匹配全称
    text = clean_place_text(location)

    if not text:
        return set()

    # 2. 保留原始大小写，提取独立单词
    raw_tokens = set(
        re.split(r"[,;|\s]+", str(location).strip())
    )

    matches = set()

    # 3. 识别美国、加拿大的缩写和全称
    for mapping, country in [
        (us_states, "United States"),
        (canada_states, "Canada"),
    ]:
        for abbreviation, full_name in mapping.items():
            abbreviation_match = abbreviation in raw_tokens
            full_name_match = contains_place(text, full_name)

            if abbreviation_match or full_name_match:
                matches.add((full_name, country))

    # 4. 原作者的澳大利亚字典提供州名全称
    for full_name in australia_states:
        if contains_place(text, full_name):
            matches.add((full_name, "Australia"))

    return matches


def canonical_state(state, country):
    if pd.isna(state):
        return ""

    state = str(state).strip()
    country = clean_place_text(country)

    if country == "united states":
        return us_states.get(state.upper(), state)

    if country == "canada":
        return canada_states.get(state.upper(), state)

    return state


def advanced_geo_feature(cddts, countries, city_to_places):
    # 1. 每个用户只解析一次位置
    user_places = {}

    users = cddts[
        ["user", "location"]
    ].drop_duplicates("user")

    for row in users.itertuples(index=False):
        user_places[row.user] = parse_location(
            row.location,
            countries,
            city_to_places,
        )

    # 2. 逐候选比较活动地点
    scores = []

    for row in cddts.itertuples(index=False):
        event_country = clean_place_text(row.country)

        if event_country == "democratic republic congo":
            event_country = "democratic republic of the congo"

        event_state = clean_place_text(
            canonical_state(row.state, row.country)
        )
        event_city = clean_place_text(row.city)

        best_score = 0.0

        for place in user_places[row.user]:
            user_country = clean_place_text(
                place.get("country", "")
            )

            # 国家已知且一致，才继续比较州和城市
            if not event_country or event_country != user_country:
                continue

            user_state = clean_place_text(
                place.get("state", "")
            )
            user_city = clean_place_text(
                place.get("city", "")
            )

            state_match = bool(
                event_state and event_state == user_state
            )
            city_match = bool(
                event_city and event_city == user_city
            )

            score = 1 + state_match + 2 * city_match
            best_score = max(best_score, score)

        scores.append(best_score)

    features = pd.DataFrame(index=cddts.index)
    features["advanced_location_score"] = scores

    print("增强地理匹配特征生成结束")
    return features