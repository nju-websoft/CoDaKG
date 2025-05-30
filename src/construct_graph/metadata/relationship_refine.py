import csv
import Levenshtein
import re
import pymysql
import pandas as pd

def normalizeText(text):
    lowercaseText = text.lower()
    return "".join(lowercaseText.split())


def prefixMatchSuffixDiff(d1Name, d2Name):
    delimiterList = [",by ", "- ", ": ", ', ']
    for delimiter in delimiterList:
        d1Split = d1Name.split(delimiter)
        d2Split = d2Name.split(delimiter)
        if len(d1Split) == 2 and len(d2Split) == 2 and (
                Levenshtein.distance(normalizeText(d1Split[0]), normalizeText(d2Split[0])) / max(
                len(normalizeText(d1Split[0])), len(normalizeText(d2Split[0]))) < 0.2) and normalizeText(
                d1Split[1]) != normalizeText(d2Split[1]):
            return True
    return False


def prefixDiffSuffixMatch(d1Name, d2Name):
    delimiterList = [",by ", "- ", ": ", ', ']
    for delimiter in delimiterList:
        d1Split = d1Name.split(delimiter)
        d2Split = d2Name.split(delimiter)
        if len(d1Split) == 2 and len(d2Split) == 2 and (normalizeText(d1Split[0]) != normalizeText(d2Split[0])) and (
                Levenshtein.distance(normalizeText(d1Split[1]), normalizeText(d2Split[1])) / max(
                len(normalizeText(d1Split[1])), len(normalizeText(d2Split[1]))) < 0.2):
            return True
    return False


def removeVersionNumber(text):
    versionRegex = r"\W[V,v](\s)?[\.\d]{1,10}|\W[v,V]ersion(\s)?[\.\d]{1,10}"
    new_text = re.sub(versionRegex, '', text)
    return new_text


def removeDate(text):
    dateRegex = r"(?:[0-9]{2})?[0-9]{2}-[0-3]?[0-9]-[0-3]?[0-9]"
    new_text = re.sub(dateRegex, '', text)
    yearRegex = r"[,:-][\s]?\d{4,}"
    new_text = re.sub(yearRegex, '', new_text)
    # print(new_text)
    return new_text


def isReplica(d1, d2):
    d1Info = normalizeText(d1)
    d2Info = normalizeText(d2)
    return Levenshtein.distance(d1Info, d2Info) / max(len(d1Info), len(d2Info)) < 0.1 or d1Info.startswith(
        d2Info) or d2Info.startswith(d1Info)


def isVariant(d1, d2):
    d1Name = normalizeText(removeVersionNumber(removeDate(d1)))
    d2Name = normalizeText(removeVersionNumber(removeDate(d2)))
    return Levenshtein.distance(d1Name, d2Name) / max(len(d1Name), len(d2Name)) < 0.3 or prefixMatchSuffixDiff(d1,
                                                                                                               d2) or prefixDiffSuffixMatch(
        d1, d2)

def isVersion(d1, d2):
    new_d1 = removeVersionNumber(d1)
    new_d2 = removeVersionNumber(d2)
    is_ver_exist = False
    if normalizeText(new_d1) != normalizeText(d1) or normalizeText(new_d2) != normalizeText(d2):
        is_ver_exist = True
    if normalizeText(new_d1) == normalizeText(new_d2) and is_ver_exist:
        return True
    return False

relationship_type = ['NULL', 'Replica', 'Variant', 'Subset', 'Derived', 'Version']

user = "xtpan"
pwd = "panxintian123"

conn = pymysql.connect(
    host="114.212.189.98", user=user, password=pwd, database="dataset_kg_2025mar" # ,cursorclass=pymysql.cursors.DictCursor
)
cursor = conn.cursor()

sql = "SELECT dataset_id, title, description from acordar_datasets"

cursor.execute(sql)

responses = cursor.fetchall()

cursor.close()
conn.close()

text_dict = {}

for response in responses:
    text_dict[response[0]] = {"title": response[1], "description": response[2]}

relationships = []
# Open the CSV file. Replace 'file.csv' with your filename.
with open('metadata/pairs_relationships_acordar.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    
    # Optionally, if your CSV file has a header, skip it:
    header = next(csvreader, None)
    
    # Loop through each row in the CSV file
    for row in csvreader:
        # Assuming the CSV columns are in the order:
        # dataset_id1, dataset_id2, relationship
        dataset_id1 = row[0]
        dataset_id2 = row[1]
        ind = int(row[2])
        name1 = text_dict[dataset_id1]['title']
        name2 = text_dict[dataset_id2]['title']
        
        # Process the row (here we simply print the values)

        if name1 == name2 or dataset_id1 == dataset_id2:
            relationships.append([dataset_id1, dataset_id2, 'Replica'])
            continue
        if isVersion(removeDate(name1), removeDate(name2)):
            relationships.append([dataset_id1, dataset_id2, 'Version'])
            continue
        if isVariant(name1, name2):
            relationships.append([dataset_id1, dataset_id2, 'Variant'])
        elif ind == 1:
            if isReplica(name1, name2):
                relationships.append([dataset_id1, dataset_id2, 'Replica'])
        else:
            relationships.append([dataset_id1, dataset_id2, relationship_type[ind]])

relationships_df = pd.DataFrame(relationships, columns=['id1', 'id2', 'relationship'])
relationships_df.to_csv('refine_relationships_acordar.csv', index=False)