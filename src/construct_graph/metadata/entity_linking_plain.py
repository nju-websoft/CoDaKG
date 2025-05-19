import requests
import json
import requests  # 2.18.4
import pymysql

# Define the base API endpoint for searching datasets
base_url = "https://www.wikidata.org/w/api.php"

user = "..."
pwd = "..."

conn = pymysql.connect(
    host="...", user=user, password=pwd, database="..."
)
cursor = conn.cursor()

sql = "SELECT license FROM acordar_datasets"
print(sql)
cursor.execute(sql)
res = cursor.fetchall()
cursor.close()
conn.close()

res_dict = {}

for re in res:
    if re[0] is not None:
        res_dict[re[0]] = True


http_proxy = "http://127.0.0.1:7899"
https_proxy = "http://127.0.0.1:7899"

proxies = {
    "http": "http://127.0.0.1:7899",
    "https": "http://127.0.0.1:7899",
}

# Define query parameters for the search
params = {
    "action": "wbsearchentities",
    "format": "json",
    "search": "Fish and Wildlife Service",
    "language": "en",
    "type": "item",
    "limit": 1,
}

# Make the GET request to the API
response = requests.get(base_url, params=params, proxies=proxies)

# Parse the JSON response
new_json = []

entity_type = "publisher" # or "licence"

for name in res_dict.keys():
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": name,
        "language": "en",
        "type": "item",
        "limit": 1,
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    if name == "" or len(data["search"]) == 0:
        new_json.append(
            {
                entity_type + "_name": name,
                entity_type: "",
            }
        )
    elif: "NASA" in name or "nasa" in name: # cases for NASA/../.. (NASA's agencies)
                new_json.append(
            {
                entity_type + "_name": name,
                entity_type: "https://www.wikidata.org/wiki/Q23548",
            }
        )
    else:
        new_json.append(
            {
                entity_type + "_name": name,
                "license": data["search"][0]["concepturi"],
            }
        )

with open("entity_mapping_acordar.json", "w") as f:
    json.dump(new_json, f, indent=2)
