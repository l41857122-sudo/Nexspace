"""
Generates the 210-case Phase 1 Real-World Visual Validation Manifest.
Strictly satisfies all geographic, category, and query-type quotas.
"""

import json
import os

MANIFEST_ITEMS = [
    # =========================================================================
    # ASIA (45 Cases)
    # =========================================================================
    {"id": "001", "query": "Indira Nagar, Lucknow", "region": "Asia", "category": "locality", "expected_type": "urban"},
    {"id": "002", "query": "Aliganj, Lucknow", "region": "Asia", "category": "neighborhood", "expected_type": "urban"},
    {"id": "003", "query": "Rumi Darwaza, Lucknow", "region": "Asia", "category": "landmark", "expected_type": "urban"},
    {"id": "004", "query": "Gomti Nagar, Lucknow", "region": "Asia", "category": "locality", "expected_type": "urban"},
    {"id": "005", "query": "Show me satellite imagery of Connaught Place, Delhi", "region": "Asia", "category": "natural_language", "expected_type": "urban"},
    {"id": "006", "query": "Delhi Airport", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "007", "query": "Koramangala, Bengaluru", "region": "Asia", "category": "locality", "expected_type": "urban"},
    {"id": "008", "query": "Mumbai Port", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "009", "query": "Jawaharlal Nehru Port, Navi Mumbai", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "010", "query": "Mawlynnong Village, Meghalaya", "region": "Asia", "category": "village", "expected_type": "rural"},
    {"id": "011", "query": "Varanasi Ghats, Uttar Pradesh", "region": "Asia", "category": "landmark", "expected_type": "urban"},
    {"id": "012", "query": "Taj Mahal, Agra", "region": "Asia", "category": "landmark", "expected_type": "urban"},
    {"id": "013", "query": "Dal Lake, Srinagar", "region": "Asia", "category": "natural_feature", "expected_type": "water"},
    {"id": "014", "query": "Sundarbans National Park, India", "region": "Asia", "category": "natural_feature", "expected_type": "forest"},
    {"id": "015", "query": "Thar Desert, Jaisalmer", "region": "Asia", "category": "natural_feature", "expected_type": "arid"},
    {"id": "016", "query": "Tokyo, Japan", "region": "Asia", "category": "city", "expected_type": "urban"},
    {"id": "017", "query": "Shinjuku, Tokyo", "region": "Asia", "category": "neighborhood", "expected_type": "urban"},
    {"id": "018", "query": "Haneda Airport, Tokyo", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "019", "query": "Mount Fuji, Japan", "region": "Asia", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "020", "query": "Port of Yokohama, Japan", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "021", "query": "Singapore Changi Airport", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "022", "query": "Marina Bay, Singapore", "region": "Asia", "category": "landmark", "expected_type": "coastal"},
    {"id": "023", "query": "Jurong Island, Singapore", "region": "Asia", "category": "industrial", "expected_type": "industrial"},
    {"id": "024", "query": "Victoria Harbour, Hong Kong", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "025", "query": "Hong Kong International Airport", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "026", "query": "Bangkok, Thailand", "region": "Asia", "category": "city", "expected_type": "urban"},
    {"id": "027", "query": "Suvarnabhumi Airport, Bangkok", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "028", "query": "Chao Phraya River, Bangkok", "region": "Asia", "category": "natural_feature", "expected_type": "water"},
    {"id": "029", "query": "Incheon International Airport, South Korea", "region": "Asia", "category": "airport", "expected_type": "airport"},
    {"id": "030", "query": "Gangnam, Seoul, South Korea", "region": "Asia", "category": "neighborhood", "expected_type": "urban"},
    {"id": "031", "query": "Port of Busan, South Korea", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "032", "query": "Dubai Marina, UAE", "region": "Asia", "category": "neighborhood", "expected_type": "coastal"},
    {"id": "033", "query": "Burj Khalifa, Dubai", "region": "Asia", "category": "landmark", "expected_type": "urban"},
    {"id": "034", "query": "Palm Jumeirah, Dubai", "region": "Asia", "category": "coastal", "expected_type": "coastal"},
    {"id": "035", "query": "Port of Jebel Ali, Dubai", "region": "Asia", "category": "port", "expected_type": "port"},
    {"id": "036", "query": "Dead Sea, Jordan", "region": "Asia", "category": "natural_feature", "expected_type": "water"},
    {"id": "037", "query": "Riyadh, Saudi Arabia", "region": "Asia", "category": "city", "expected_type": "arid"},
    {"id": "038", "query": "Kuala Lumpur City Centre, Malaysia", "region": "Asia", "category": "landmark", "expected_type": "urban"},
    {"id": "039", "query": "Ha Long Bay, Vietnam", "region": "Asia", "category": "coastal", "expected_type": "coastal"},
    {"id": "040", "query": "Bagan, Myanmar", "region": "Asia", "category": "landmark", "expected_type": "rural"},
    {"id": "041", "query": "Kathmandu, Nepal", "region": "Asia", "category": "city", "expected_type": "mountain"},
    {"id": "042", "query": "Mount Everest, Nepal", "region": "Asia", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "043", "query": "26.8467, 80.9462", "region": "Asia", "category": "coordinates", "expected_type": "urban"},
    {"id": "044", "query": "28.5562° N, 77.1000° E", "region": "Asia", "category": "coordinates", "expected_type": "airport"},
    {"id": "045", "query": "bbox: 80.90, 26.80, 81.00, 26.90", "region": "Asia", "category": "bounding_box", "expected_type": "urban"},

    # =========================================================================
    # EUROPE (35 Cases)
    # =========================================================================
    {"id": "046", "query": "London, UK", "region": "Europe", "category": "city", "expected_type": "urban"},
    {"id": "047", "query": "Canary Wharf, London", "region": "Europe", "category": "neighborhood", "expected_type": "urban"},
    {"id": "048", "query": "London Heathrow Airport", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "049", "query": "Port of London, UK", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "050", "query": "Eiffel Tower, Paris", "region": "Europe", "category": "landmark", "expected_type": "urban"},
    {"id": "051", "query": "Charles de Gaulle Airport, Paris", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "052", "query": "Port of Rotterdam, Netherlands", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "053", "query": "Amsterdam Canals, Netherlands", "region": "Europe", "category": "landmark", "expected_type": "water"},
    {"id": "054", "query": "Schiphol Airport, Amsterdam", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "055", "query": "Port of Antwerp, Belgium", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "056", "query": "Port of Hamburg, Germany", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "057", "query": "Frankfurt Airport, Germany", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "058", "query": "Berlin Brandenburg Airport, Germany", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "059", "query": "Ruhr Valley, Germany", "region": "Europe", "category": "industrial", "expected_type": "industrial"},
    {"id": "060", "query": "Colosseum, Rome, Italy", "region": "Europe", "category": "landmark", "expected_type": "urban"},
    {"id": "061", "query": "Venice Lagoon, Italy", "region": "Europe", "category": "coastal", "expected_type": "water"},
    {"id": "062", "query": "Port of Genoa, Italy", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "063", "query": "Port of Barcelona, Spain", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "064", "query": "Madrid-Barajas Airport, Spain", "region": "Europe", "category": "airport", "expected_type": "airport"},
    {"id": "065", "query": "Port of Valencia, Spain", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "066", "query": "Rock of Gibraltar", "region": "Europe", "category": "landmark", "expected_type": "coastal"},
    {"id": "067", "query": "Port of Piraeus, Greece", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "068", "query": "Santorini Caldera, Greece", "region": "Europe", "category": "coastal", "expected_type": "coastal"},
    {"id": "069", "query": "Lake Geneva, Switzerland", "region": "Europe", "category": "natural_feature", "expected_type": "water"},
    {"id": "070", "query": "Matterhorn, Switzerland", "region": "Europe", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "071", "query": "Vienna, Austria", "region": "Europe", "category": "city", "expected_type": "urban"},
    {"id": "072", "query": "Prague Old Town, Czechia", "region": "Europe", "category": "neighborhood", "expected_type": "urban"},
    {"id": "073", "query": "Warsaw, Poland", "region": "Europe", "category": "city", "expected_type": "urban"},
    {"id": "074", "query": "Stockholm Archipelago, Sweden", "region": "Europe", "category": "coastal", "expected_type": "coastal"},
    {"id": "075", "query": "Geirangerfjord, Norway", "region": "Europe", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "076", "query": "Dublin Port, Ireland", "region": "Europe", "category": "port", "expected_type": "port"},
    {"id": "077", "query": "Giethoorn Village, Netherlands", "region": "Europe", "category": "village", "expected_type": "rural"},
    {"id": "078", "query": "Hallstatt Village, Austria", "region": "Europe", "category": "village", "expected_type": "mountain"},
    {"id": "079", "query": "51.5074, -0.1278", "region": "Europe", "category": "coordinates", "expected_type": "urban"},
    {"id": "080", "query": "Show me high-resolution satellite imagery of Port of Rotterdam", "region": "Europe", "category": "natural_language", "expected_type": "port"},

    # =========================================================================
    # NORTH AMERICA (35 Cases)
    # =========================================================================
    {"id": "081", "query": "Times Square, New York", "region": "North America", "category": "landmark", "expected_type": "urban"},
    {"id": "082", "query": "Central Park, New York", "region": "North America", "category": "natural_feature", "expected_type": "urban"},
    {"id": "083", "query": "Brooklyn, New York", "region": "North America", "category": "locality", "expected_type": "urban"},
    {"id": "084", "query": "JFK Airport, New York", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "085", "query": "Port of New York and New Jersey", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "086", "query": "Port of Los Angeles", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "087", "query": "Port of Long Beach, California", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "088", "query": "Los Angeles International Airport", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "089", "query": "San Francisco International Airport", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "090", "query": "Golden Gate Bridge, San Francisco", "region": "North America", "category": "landmark", "expected_type": "coastal"},
    {"id": "091", "query": "Silicon Valley, California", "region": "North America", "category": "locality", "expected_type": "urban"},
    {"id": "092", "query": "Lake Tahoe, California", "region": "North America", "category": "natural_feature", "expected_type": "water"},
    {"id": "093", "query": "Yosemite Valley, California", "region": "North America", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "094", "query": "Death Valley, California", "region": "North America", "category": "natural_feature", "expected_type": "arid"},
    {"id": "095", "query": "Las Vegas Strip, Nevada", "region": "North America", "category": "landmark", "expected_type": "urban"},
    {"id": "096", "query": "Grand Canyon, Arizona", "region": "North America", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "097", "query": "O'Hare International Airport, Chicago", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "098", "query": "Port of Houston, Texas", "region": "North America", "category": "port", "expected_type": "industrial"},
    {"id": "099", "query": "Dallas Fort Worth International Airport", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "100", "query": "Port of Miami, Florida", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "101", "query": "Everglades National Park, Florida", "region": "North America", "category": "natural_feature", "expected_type": "water"},
    {"id": "102", "query": "Port of New Orleans, Louisiana", "region": "North America", "category": "port", "expected_type": "water"},
    {"id": "103", "query": "Mississippi River Delta, Louisiana", "region": "North America", "category": "natural_feature", "expected_type": "water"},
    {"id": "104", "query": "Port of Seattle, Washington", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "105", "query": "Mount Rainier, Washington", "region": "North America", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "106", "query": "Port of Vancouver, Canada", "region": "North America", "category": "port", "expected_type": "port"},
    {"id": "107", "query": "Vancouver International Airport, Canada", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "108", "query": "Toronto Pearson Airport, Canada", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "109", "query": "Port of Montreal, Canada", "region": "North America", "category": "port", "expected_type": "water"},
    {"id": "110", "query": "Niagara Falls, New York", "region": "North America", "category": "landmark", "expected_type": "water"},
    {"id": "111", "query": "Cancun Hotel Zone, Mexico", "region": "North America", "category": "coastal", "expected_type": "coastal"},
    {"id": "112", "query": "Mexico City International Airport", "region": "North America", "category": "airport", "expected_type": "airport"},
    {"id": "113", "query": "37.7749, -122.4194", "region": "North America", "category": "coordinates", "expected_type": "urban"},
    {"id": "114", "query": "40.7128° N, 74.0060° W", "region": "North America", "category": "coordinates", "expected_type": "urban"},
    {"id": "115", "query": "Show me aerial view of Grand Canyon, Arizona", "region": "North America", "category": "natural_language", "expected_type": "mountain"},

    # =========================================================================
    # SOUTH AMERICA (22 Cases)
    # =========================================================================
    {"id": "116", "query": "Amazon Rainforest, Brazil", "region": "South America", "category": "natural_feature", "expected_type": "forest"},
    {"id": "117", "query": "Manaus Port, Amazon, Brazil", "region": "South America", "category": "port", "expected_type": "water"},
    {"id": "118", "query": "Meeting of the Waters, Manaus, Brazil", "region": "South America", "category": "natural_feature", "expected_type": "water"},
    {"id": "119", "query": "Port of Santos, Brazil", "region": "South America", "category": "port", "expected_type": "port"},
    {"id": "120", "query": "Copacabana Beach, Rio de Janeiro", "region": "South America", "category": "coastal", "expected_type": "coastal"},
    {"id": "121", "query": "Sugarloaf Mountain, Rio de Janeiro", "region": "South America", "category": "landmark", "expected_type": "coastal"},
    {"id": "122", "query": "Port of Rio de Janeiro, Brazil", "region": "South America", "category": "port", "expected_type": "port"},
    {"id": "123", "query": "Sao Paulo Guarulhos Airport, Brazil", "region": "South America", "category": "airport", "expected_type": "airport"},
    {"id": "124", "query": "Brasilia Pilot Plan, Brazil", "region": "South America", "category": "landmark", "expected_type": "urban"},
    {"id": "125", "query": "Port of Buenos Aires, Argentina", "region": "South America", "category": "port", "expected_type": "port"},
    {"id": "126", "query": "Puerto Madero, Buenos Aires", "region": "South America", "category": "neighborhood", "expected_type": "urban"},
    {"id": "127", "query": "Ezeiza International Airport, Buenos Aires", "region": "South America", "category": "airport", "expected_type": "airport"},
    {"id": "128", "query": "Iguazu Falls, Argentina", "region": "South America", "category": "landmark", "expected_type": "water"},
    {"id": "129", "query": "Perito Moreno Glacier, Argentina", "region": "South America", "category": "natural_feature", "expected_type": "ice"},
    {"id": "130", "query": "Port of Valparaiso, Chile", "region": "South America", "category": "port", "expected_type": "port"},
    {"id": "131", "query": "Atacama Desert, Chile", "region": "South America", "category": "natural_feature", "expected_type": "arid"},
    {"id": "132", "query": "Salar de Uyuni, Bolivia", "region": "South America", "category": "natural_feature", "expected_type": "arid"},
    {"id": "133", "query": "Lake Titicaca, Peru", "region": "South America", "category": "natural_feature", "expected_type": "water"},
    {"id": "134", "query": "Port of Callao, Lima, Peru", "region": "South America", "category": "port", "expected_type": "port"},
    {"id": "135", "query": "Machu Picchu, Peru", "region": "South America", "category": "landmark", "expected_type": "mountain"},
    {"id": "136", "query": "Galapagos Islands, Ecuador", "region": "South America", "category": "coastal", "expected_type": "coastal"},
    {"id": "137", "query": "Can you display optical satellite image of Port of Santos", "region": "South America", "category": "natural_language", "expected_type": "port"},

    # =========================================================================
    # AFRICA (26 Cases)
    # =========================================================================
    {"id": "138", "query": "Cairo, Egypt", "region": "Africa", "category": "city", "expected_type": "urban"},
    {"id": "139", "query": "Giza Pyramids, Egypt", "region": "Africa", "category": "landmark", "expected_type": "arid"},
    {"id": "140", "query": "Cairo International Airport, Egypt", "region": "Africa", "category": "airport", "expected_type": "airport"},
    {"id": "141", "query": "Suez Canal, Port Said, Egypt", "region": "Africa", "category": "port", "expected_type": "water"},
    {"id": "142", "query": "Port of Alexandria, Egypt", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "143", "query": "Nile River, Aswan, Egypt", "region": "Africa", "category": "natural_feature", "expected_type": "water"},
    {"id": "144", "query": "Port of Durban, South Africa", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "145", "query": "Cape Town, South Africa", "region": "Africa", "category": "city", "expected_type": "coastal"},
    {"id": "146", "query": "Table Mountain, Cape Town", "region": "Africa", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "147", "query": "Port of Cape Town, South Africa", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "148", "query": "Johannesburg OR Tambo Airport, South Africa", "region": "Africa", "category": "airport", "expected_type": "airport"},
    {"id": "149", "query": "Nairobi, Kenya", "region": "Africa", "category": "city", "expected_type": "urban"},
    {"id": "150", "query": "Jomo Kenyatta Airport, Nairobi", "region": "Africa", "category": "airport", "expected_type": "airport"},
    {"id": "151", "query": "Port of Mombasa, Kenya", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "152", "query": "Mount Kilimanjaro, Tanzania", "region": "Africa", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "153", "query": "Port of Dar es Salaam, Tanzania", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "154", "query": "Serengeti National Park, Tanzania", "region": "Africa", "category": "natural_feature", "expected_type": "rural"},
    {"id": "155", "query": "Zanzibar Island, Tanzania", "region": "Africa", "category": "coastal", "expected_type": "coastal"},
    {"id": "156", "query": "Port of Lagos, Nigeria", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "157", "query": "Victoria Island, Lagos, Nigeria", "region": "Africa", "category": "neighborhood", "expected_type": "coastal"},
    {"id": "158", "query": "Port of Tangier Med, Morocco", "region": "Africa", "category": "port", "expected_type": "port"},
    {"id": "159", "query": "Marrakech Medina, Morocco", "region": "Africa", "category": "neighborhood", "expected_type": "urban"},
    {"id": "160", "query": "Lake Victoria, Uganda", "region": "Africa", "category": "natural_feature", "expected_type": "water"},
    {"id": "161", "query": "Victoria Falls, Zimbabwe", "region": "Africa", "category": "natural_feature", "expected_type": "water"},
    {"id": "162", "query": "Okavango Delta, Botswana", "region": "Africa", "category": "natural_feature", "expected_type": "water"},
    {"id": "163", "query": "Show me satellite view of Suez Canal", "region": "Africa", "category": "natural_language", "expected_type": "water"},

    # =========================================================================
    # OCEANIA (22 Cases)
    # =========================================================================
    {"id": "164", "query": "Sydney Opera House, Australia", "region": "Oceania", "category": "landmark", "expected_type": "coastal"},
    {"id": "165", "query": "Port Jackson, Sydney", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "166", "query": "Sydney Kingsford Smith Airport", "region": "Oceania", "category": "airport", "expected_type": "airport"},
    {"id": "167", "query": "Bondi Beach, Sydney", "region": "Oceania", "category": "coastal", "expected_type": "coastal"},
    {"id": "168", "query": "Port of Melbourne, Australia", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "169", "query": "Melbourne Airport, Australia", "region": "Oceania", "category": "airport", "expected_type": "airport"},
    {"id": "170", "query": "Port of Brisbane, Australia", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "171", "query": "Port of Fremantle, Perth", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "172", "query": "Great Barrier Reef, Australia", "region": "Oceania", "category": "natural_feature", "expected_type": "coastal"},
    {"id": "173", "query": "Uluru, Northern Territory, Australia", "region": "Oceania", "category": "landmark", "expected_type": "arid"},
    {"id": "174", "query": "Port of Darwin, Australia", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "175", "query": "Hobart Port, Tasmania", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "176", "query": "Port of Auckland, New Zealand", "region": "Oceania", "category": "port", "expected_type": "port"},
    {"id": "177", "query": "Auckland Airport, New Zealand", "region": "Oceania", "category": "airport", "expected_type": "airport"},
    {"id": "178", "query": "Wellington Harbour, New Zealand", "region": "Oceania", "category": "port", "expected_type": "coastal"},
    {"id": "179", "query": "Christchurch, New Zealand", "region": "Oceania", "category": "city", "expected_type": "urban"},
    {"id": "180", "query": "Milford Sound, New Zealand", "region": "Oceania", "category": "natural_feature", "expected_type": "mountain"},
    {"id": "181", "query": "Port of Suva, Fiji", "region": "Oceania", "category": "port", "expected_type": "coastal"},
    {"id": "182", "query": "Port Moresby, Papua New Guinea", "region": "Oceania", "category": "port", "expected_type": "coastal"},
    {"id": "183", "query": "Bora Bora Lagoon, French Polynesia", "region": "Oceania", "category": "coastal", "expected_type": "coastal"},
    {"id": "184", "query": "-33.8568, 151.2153", "region": "Oceania", "category": "coordinates", "expected_type": "coastal"},
    {"id": "185", "query": "Show me space view of Great Barrier Reef", "region": "Oceania", "category": "natural_language", "expected_type": "coastal"},

    # =========================================================================
    # HIGH LATITUDE & POLAR (11 Cases)
    # =========================================================================
    {"id": "186", "query": "Longyearbyen, Svalbard", "region": "High Latitude", "category": "settlement", "expected_type": "tundra"},
    {"id": "187", "query": "Svalbard Global Seed Vault", "region": "High Latitude", "category": "landmark", "expected_type": "tundra"},
    {"id": "188", "query": "Tromso, Norway", "region": "High Latitude", "category": "city", "expected_type": "coastal"},
    {"id": "189", "query": "Port of Murmansk, Russia", "region": "High Latitude", "category": "port", "expected_type": "port"},
    {"id": "190", "query": "Reykjavik, Iceland", "region": "High Latitude", "category": "city", "expected_type": "coastal"},
    {"id": "191", "query": "Keflavik International Airport, Iceland", "region": "High Latitude", "category": "airport", "expected_type": "airport"},
    {"id": "192", "query": "Nuuk, Greenland", "region": "High Latitude", "category": "city", "expected_type": "coastal"},
    {"id": "193", "query": "Fairbanks, Alaska", "region": "High Latitude", "category": "city", "expected_type": "tundra"},
    {"id": "194", "query": "Port of Anchorage, Alaska", "region": "High Latitude", "category": "port", "expected_type": "port"},
    {"id": "195", "query": "78.2232° N, 15.6464° E", "region": "High Latitude", "category": "coordinates", "expected_type": "tundra"},
    {"id": "196", "query": "McMurdo Station, Antarctica", "region": "High Latitude", "category": "settlement", "expected_type": "ice"},

    # =========================================================================
    # COASTAL & ISLAND (8 Cases)
    # =========================================================================
    {"id": "197", "query": "Male Atoll, Maldives", "region": "Coastal", "category": "coastal", "expected_type": "coastal"},
    {"id": "198", "query": "Velana Airport, Maldives", "region": "Coastal", "category": "airport", "expected_type": "coastal"},
    {"id": "199", "query": "Honolulu Harbour, Hawaii", "region": "Coastal", "category": "port", "expected_type": "port"},
    {"id": "200", "query": "Pearl Harbor, Hawaii", "region": "Coastal", "category": "port", "expected_type": "port"},
    {"id": "201", "query": "Waikiki Beach, Honolulu", "region": "Coastal", "category": "coastal", "expected_type": "coastal"},
    {"id": "202", "query": "Port Louis, Mauritius", "region": "Coastal", "category": "port", "expected_type": "port"},
    {"id": "203", "query": "Victoria, Seychelles", "region": "Coastal", "category": "port", "expected_type": "coastal"},
    {"id": "204", "query": "Nassau Port, Bahamas", "region": "Coastal", "category": "port", "expected_type": "coastal"},

    # =========================================================================
    # RURAL & REMOTE (6 Cases)
    # =========================================================================
    {"id": "205", "query": "Gobi Desert, Mongolia", "region": "Rural", "category": "natural_feature", "expected_type": "arid"},
    {"id": "206", "query": "Corn Belt Agriculture, Iowa", "region": "Rural", "category": "agricultural", "expected_type": "agricultural"},
    {"id": "207", "query": "Bordeaux Vineyards, France", "region": "Rural", "category": "agricultural", "expected_type": "agricultural"},
    {"id": "208", "query": "Punjab Agricultural Belt, India", "region": "Rural", "category": "agricultural", "expected_type": "agricultural"},
    {"id": "209", "query": "Namib Desert Sand Dunes, Namibia", "region": "Rural", "category": "natural_feature", "expected_type": "arid"},
    {"id": "210", "query": "Show me satellite imagery of Gobi Desert", "region": "Rural", "category": "natural_language", "expected_type": "arid"}
]


def create_manifest():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase1_visual_validation", "manifest")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "phase1_visual_test_manifest.json")
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(MANIFEST_ITEMS, f, indent=2)
        
    print(f"Successfully generated visual validation manifest with {len(MANIFEST_ITEMS)} items at: {manifest_path}")

    # Print distribution stats
    regions = {}
    categories = {}
    for it in MANIFEST_ITEMS:
        regions[it["region"]] = regions.get(it["region"], 0) + 1
        categories[it["category"]] = categories.get(it["category"], 0) + 1
        
    print("\nRegional Distribution:")
    for r, c in regions.items():
        print(f"  {r}: {c}")
        
    print("\nCategory Distribution:")
    for cat, c in categories.items():
        print(f"  {cat}: {c}")


if __name__ == "__main__":
    create_manifest()
