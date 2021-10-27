import pandas as pd
import requests
import logging
from retrying import retry
import networkx as nx
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


API_URL_WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
API_URL_WIKIDATA = "https://query.wikidata.org/sparql"
SESSION = None


def get_session(session=None):
    """Get session object
    Args:
        session (requests.Session): request session object
    Returns:
        requests.Session: request session object
    """

    if session is None:
        global SESSION
        if SESSION is None:
            SESSION = requests.Session()
        session = SESSION

    return session

@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def find_wikidata_id(name, limit=1, session=None):
    """Find the entity ID in wikidata from a title string.
    Args:
        name (str): A string with search terms (eg. "Batman (1989) film")
        limit (int): Number of results to return
        session (requests.Session): requests session to reuse connections
    Returns:
        str: wikidata entityID corresponding to the title string. 'entityNotFound' will be returned if no page is found
    """

    session = get_session(session=session)

    params = dict(
        action="query",
        list="search",
        srsearch=bytes(name, encoding="utf8"),
        srlimit=limit,
        srprop="",
        format="json",
    )

    try:
        response = session.get(API_URL_WIKIPEDIA, params=params)
        page_id = response.json()["query"]["search"][0]["pageid"]
    except Exception as e:
        # TODO: distinguish between connection error and entity not found
        # logger.error("ENTITY NOT FOUND")
        return "entityNotFound", "entityNotFound"

    params = dict(
        action="query",
        prop="pageprops",
        ppprop="wikibase_item",
        pageids=[page_id],
        format="json",
    )

    try:
        response = session.get(API_URL_WIKIPEDIA, params=params)
        entity_id = response.json()["query"]["pages"][str(page_id)]["pageprops"][
            "wikibase_item"
        ]
        entity_name = response.json()["query"]["pages"][str(page_id)]["title"]
    except Exception as e:
        # TODO: distinguish between connection error and entity not found
        # logger.error("ENTITY NOT FOUND")
        return "entityNotFound", "entityNotFound"

    return entity_id, entity_name


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def query_entity_links(entity_id, limit=500, session=None):
    """Query all linked pages from a wikidata entityID
    Args:
        entity_id (str): A wikidata entity ID
        session (requests.Session): requests session to reuse connections
    Returns:
        json: Dictionary with linked pages.
    """
    query = (
        """
    PREFIX entity: <http://www.wikidata.org/entity/>
    #partial results
    SELECT ?propUrl ?propLabel ?valUrl ?valLabel
    WHERE
    {
        hint:Query hint:optimizer 'None' .
        {	BIND(entity:"""
        + entity_id
        + """ AS ?valUrl) .
            BIND("N/A" AS ?propUrl ) .
            BIND("identity"@en AS ?propLabel ) .
        }
        UNION
        {	entity:"""
        + entity_id
        + """ ?propUrl ?valUrl .
            ?property ?ref ?propUrl .
            ?property rdf:type wikibase:Property .
            ?property rdfs:label ?propLabel
        }
        ?valUrl rdfs:label ?valLabel
        FILTER (LANG(?valLabel) = 'en') .
        OPTIONAL{ ?valUrl wdt:P18 ?picture .}
        FILTER (lang(?propLabel) = 'en' )
    }
    ORDER BY ?propUrl ?valUrl
    LIMIT 
    """+str(limit)
    )

    session = get_session(session=session)

    try:
        data = session.get(
            API_URL_WIKIDATA, params=dict(query=query, format="json")
        ).json()
    except Exception as e:
        # logger.error("ENTITY NOT FOUND")
        return {}

    return data

def read_linked_entities(data):
    """Obtain lists of liken entities (IDs and names) from dictionary
    Args:
        data (json): dictionary with linked pages
    Returns:
        list, list:
        - List of liked entityIDs.
        - List of liked entity names.
    """

    return [
        (
            c.get("valUrl").get("value").replace("http://www.wikidata.org/entity/", ""),
            c.get("valLabel").get("value"),
            c.get("propUrl").get("value").replace("http://www.wikidata.org/", "").split("/")[-1],
            c.get("propLabel").get("value"),
        )
        for c in data.get("results", {}).get("bindings", [])
    ]


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def query_entity_description(entity_id, session=None):
    """Query entity wikidata description from entityID
    Args:
        entity_id (str): A wikidata page ID.
        session (requests.Session): requests session to reuse connections
    Returns:
        str: Wikidata short description of the entityID
        descriptionNotFound' will be returned if no description is found
    """
    query = (
        """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX schema: <http://schema.org/>
    SELECT ?o
    WHERE 
    {
      wd:"""
        + entity_id
        + """ schema:description ?o.
      FILTER ( lang(?o) = "en" )
    }
    """
    )

    session = get_session(session=session)

    try:
        r = session.get(API_URL_WIKIDATA, params=dict(query=query, format="json"))
        description = r.json()["results"]["bindings"][0]["o"]["value"]
    except Exception as e:
        # logger.error("DESCRIPTION NOT FOUND")
        return "descriptionNotFound"

    return description


def search_wikidata(names, extras=None, describe=True, verbose=False):
    """Create DataFrame of Wikidata search results
    Args:
        names (list[str]): List of names to search for
        extras (dict(str: list)): Optional extra items to assign to results for corresponding name
        describe (bool): Optional flag to include description of entity
        verbose (bool): Optional flag to print out intermediate data
    Returns:
        pandas.DataFrame: Wikipedia results for all names with found entities
    """

    results = []
    for idx, name in enumerate(names):
        entity_id, entity_name = find_wikidata_id(name)
        if verbose:
            print("name: {name}, entity_id: {id}".format(name=name, id=entity_id))

        if entity_id == "entityNotFound":
            continue

        json_links = query_entity_links(entity_id)
        related_links = read_linked_entities(json_links)
        description = query_entity_description(entity_id) if describe else ""

        for related_entity, related_name in related_links:
            result = dict(
                name=name,
                original_entity=entity_id,
                linked_entities=related_entity,
                name_linked_entities=related_name,
            )
            if describe:
                result["description"] = description
            if extras is not None:
                for field, lst in extras.items():
                    result[field] = lst[idx]
            results.append(result)

    return pd.DataFrame(results)


if __name__ == '__main__':
    names = ["The Godfather", "Al Pacino", "Tom Hanks", "Forrest Gump", "Julia Roberts", "", "My Best Friend's Wedding"]
        
    results_list = []
    notFound = []
    for idx, name in enumerate(names):
        # first get the wikipedia entity_id for each name
        entity_id, entity_name = find_wikidata_id(name)
        if entity_id == "entityNotFound":
            notFound.append(name)
            continue
        
        # next we query wikipedia to get entity links
        json_links = query_entity_links(entity_id)

        # the following function extracts entities from the links
        related_links = read_linked_entities(json_links)

        # now we can construct an connection in our graph between two entities
        for related_entity, related_name, relation, relation_name  in related_links:
            result = dict(
                head=entity_id,
                head_name=name,
                relation = relation,
                relation_name= relation_name,
                tail=related_entity,
                tail_name=related_name,
            )
            results_list.append(result)

    results_list = pd.DataFrame(results_list)
    print(results_list.head())

    # visual graph
    G = nx.from_pandas_edgelist(results_list, 'head', 'tail')
    target_names = results_list[["tail", "tail_name"]].drop_duplicates().rename(columns={"tail": "labels", "tail_name": "name"})
    source_names = results_list[["head", "name"]].drop_duplicates().rename(columns={"head": "labels"})
    names = pd.concat([target_names, source_names])
    names = names.set_index("labels")
    names = names.to_dict()["name"]
    plt.figure(figsize=(12,12)) 
    pos = nx.spring_layout(G)
    nx.draw(G,pos, node_size=60,font_size=9, width = 0.2)
    nx.draw_networkx_labels(G, pos, names, font_size=9)
    plt.show()