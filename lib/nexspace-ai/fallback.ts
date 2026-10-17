import { FAQ_ENTRIES } from "./knowledge";

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * NexSpace Conversational AI Engine — V2 Fallback & Offline Engine
 * Handles casual conversation, multilingual responses (Hindi, Hinglish, Spanish, French),
 * general knowledge, coding, remote sensing, and NexSpace-specific inquiries.
 */
export function answerFromKnowledgeBase(
  message: string,
  conversation: HistoryMessage[] = []
): string {
  const query = message.trim();
  const lower = query.toLowerCase();

  // ---------------------------------------------------------------------------
  // 1. Context & Pronoun Tracking from Conversation History
  // ---------------------------------------------------------------------------
  let contextualTopic = "";
  if (conversation && conversation.length > 0) {
    const historyText = conversation
      .slice(-4)
      .map((m) => m.content.toLowerCase())
      .join(" ");

    if (historyText.includes("ndvi")) {
      contextualTopic = "ndvi";
    } else if (historyText.includes("sar") || historyText.includes("radar")) {
      contextualTopic = "sar";
    } else if (historyText.includes("grounding dino") || historyText.includes("detection")) {
      contextualTopic = "detection";
    } else if (historyText.includes("recursion")) {
      contextualTopic = "recursion";
    }
  }

  // ---------------------------------------------------------------------------
  // 2. Follow-Up Resolution (Contextual Questions)
  // ---------------------------------------------------------------------------
  if (
    contextualTopic === "ndvi" &&
    (lower.includes("why nir") ||
      lower.includes("why do we need nir") ||
      lower.includes("nir kyu") ||
      lower.includes("nir kyu chahiye") ||
      lower.includes("need nir"))
  ) {
    return (
      "We need Near-Infrared (NIR) because healthy chlorophyll in plants strongly reflects NIR radiation, " +
      "which our eyes and standard RGB sensors cannot detect. Healthy vegetation absorbs red light for photosynthesis " +
      "and reflects NIR. By calculating the ratio between NIR and Red, we can gauge true biological health and vigor."
    );
  }

  const hasPronoun = /\b(it|this|that|they|these)\b/i.test(lower);
  if (hasPronoun && contextualTopic === "ndvi" && (lower.includes("support") || lower.includes("calculate") || lower.includes("have"))) {
    return (
      "Yes, NexSpace supports NDVI calculation! It uses genuine Sentinel-2 L2A multispectral data by computing " +
      "(B08 - B04) / (B08 + B04) using the Near-Infrared and Red bands to analyze vegetation health and crop density."
    );
  }

  // ---------------------------------------------------------------------------
  // 3. Hallucination Prevention Guards (NexSpace-Specific Safeguards)
  // ---------------------------------------------------------------------------

  // SAM-2 Guard
  if (lower.includes("sam-2") || lower.includes("sam 2") || lower.includes("segment anything")) {
    return (
      "NexSpace's integration with SAM-2 (Segment Anything Model 2) is not verified and not confirmed in the codebase. " +
      "Currently, verified specialist components include Grounding DINO for open-vocabulary detection, BLIP for scene captioning, " +
      "and dedicated multispectral spectral index modules."
    );
  }

  // NDVI from RGB Photo Guard
  if (
    (lower.includes("rgb") && lower.includes("ndvi")) ||
    (lower.includes("photo") && lower.includes("ndvi")) ||
    (lower.includes("camera") && lower.includes("ndvi"))
  ) {
    return (
      "No, NexSpace cannot compute genuine NDVI from regular RGB photos. " +
      "NDVI strictly requires genuine multispectral bands — specifically Red (Band 4) and Near-Infrared (Band 8) " +
      "from satellite sensors like Sentinel-2 L2A. RGB images lack the infrared band necessary for vegetation index calculations."
    );
  }

  // Universal Satellite Constellation Guard
  if (
    lower.includes("any satellite") ||
    lower.includes("all satellite") ||
    lower.includes("planet") ||
    lower.includes("worldview") ||
    lower.includes("maxar")
  ) {
    return (
      "NexSpace does not claim universal satellite support. The platform is specifically optimized for Sentinel-2 L2A " +
      "optical and Sentinel-1 SAR imagery accessed via Element84 Earth Search and AWS Open STAC. Commercial constellations " +
      "like PlanetScope or WorldView are not currently supported out of the box."
    );
  }

  // Optical-SAR Fusion Parity Caveat
  if (
    (lower.includes("optical") && lower.includes("sar") && (lower.includes("accuracy") || lower.includes("improve") || lower.includes("better"))) ||
    lower.includes("fusion improve") ||
    lower.includes("fusion accuracy")
  ) {
    return (
      "While NexSpace supports cross-modal Optical + SAR feature fusion with CRS-aware co-registration into a shared 1536-dimensional representation, " +
      "performance parity over single-modality baselines has not been established. Cross-modal fusion remains an experimental area with specific " +
      "co-registration caveats, and should not be assumed to universally exceed single-sensor optical or SAR baselines."
    );
  }

  // ---------------------------------------------------------------------------
  // 4. Casual Conversation & Slang / Tone Matching (Hinglish & English)
  // ---------------------------------------------------------------------------

  if (lower.includes("aur bhidu") || lower.includes("bhidu kaise ho")) {
    return "Mai badiya hu bhidu 😎 Tu bata, kya scene hai?";
  }

  if (lower.includes("bhai kya chal rha") || lower.includes("bhai kya chal raha") || lower.includes("kya chal rha") || lower.includes("kya chal raha hai")) {
    return "Bas bhai, NexSpace ke saath duty pe laga hu 😂 Tu bol, kya karna hai?";
  }

  if (lower === "what's up?" || lower === "whats up" || lower === "what's up" || lower === "sup" || lower === "sup?") {
    return "All good 😎 What are you working on?";
  }

  if (lower === "hello" || lower === "hey" || lower === "hi" || lower === "heyy" || lower === "yo") {
    return "Heyy! 👋 Kya scene hai?";
  }

  if (lower.includes("kaise ho") || lower.includes("kaisa hai") || lower === "kaise ho?") {
    return "Mai badiya hu bhai 😎 Tu bata?";
  }

  if (lower.includes("kya kar rhe ho") || lower.includes("kya kar rahe ho")) {
    return "Bas NexSpace ke users ki help kar raha hu 😎 Tu bata kya chal raha?";
  }

  if (lower.includes("bore ho rha") || lower.includes("bore ho raha")) {
    return "Chal phir kuch interesting karte hain 😂 Satellite view dekhna hai ya coding ki baat karein?";
  }

  if (lower.includes("good night") || lower.includes("shubh ratri")) {
    return "Good night bhai 🌙 Kal milte hain.";
  }

  if (lower.includes("good morning") || lower.includes("shubh prabhat")) {
    return "Good morning! ☀️ Ready to explore some satellite data or code today?";
  }

  if (lower === "thanks" || lower === "thank you" || lower.includes("dhanyawad") || lower.includes("shukriya")) {
    return "Anytime bhai 🤝";
  }

  if (lower.includes("you are cool") || lower.includes("you're cool") || lower.includes("tu mast hai") || lower.includes("awesome")) {
    return "Haha thanks 😭🤝 Tu bhi kam nahi hai!";
  }

  if (lower === "how are you?" || lower === "how are you" || lower === "how are you doing?") {
    return "All good 😎 What are you working on?";
  }

  if (lower.includes("tell me a joke") || lower.includes("joke sunao")) {
    return "Why do programmers prefer dark mode? Because light attracts bugs! 😂";
  }

  // ---------------------------------------------------------------------------
  // 5. Multilingual Responses (Hindi, Spanish, French)
  // ---------------------------------------------------------------------------

  // Hindi (Devanagari)
  if (query.includes("आप कैसे हैं") || query.includes("कैसे हैं आप")) {
    return "नमस्ते! मैं बिल्कुल बढ़िया हूँ 😎 आप बताइए, आज क्या करना है?";
  }

  if (query.includes("एनडीवीआई क्या है") || query.includes("ndvi क्या है")) {
    return (
      "NDVI (Normalized Difference Vegetation Index) एक वनस्पति सूचकांक है जो उपग्रह डेटा से पौधों और फसलों के स्वास्थ्य को मापता है। " +
      "यह सेंटिनल-2 के लाल (Red Band 4) और निकट-अवरक्त (NIR Band 8) स्पेक्ट्रल बैंड का उपयोग करता है। स्वस्थ वनस्पति इन्फ्रारेड को परावर्तित करती है।"
    );
  }

  if (query.includes("यह कैसे काम करता है")) {
    return (
      "यह basically satellite से STAC के ज़रिए Sentinel-2 डेटा खोजता है, और Grounding DINO तथा multispectral AI मॉडल्स से विश्लेषण करके " +
      "सटीक निर्देशांक (coordinates) और सबूत प्रदान करता है।"
    );
  }

  // Spanish
  if (lower.includes("¿cómo estás?") || lower.includes("como estas") || lower.includes("cómo estás")) {
    return "¡Hola! Todo bien por aquí 😎 ¿En qué te puedo ayudar hoy?";
  }

  if (lower.includes("¿cómo funciona?") || lower.includes("como funciona") || lower.includes("cómo funciona")) {
    return (
      "Funciona básicamente capturando imágenes satelitales Sentinel-2 vía STAC y analizándolas con modelos avanzados " +
      "de inteligencia artificial como Grounding DINO y análisis multiespectral para entregar resultados georreferenciados."
    );
  }

  // French
  if (lower.includes("comment ça va") || lower.includes("comment ca va") || lower.includes("ça va ?")) {
    return "Bonjour ! Ça va très bien merci 😎 Comment puis-je t'aider aujourd'hui ?";
  }

  // ---------------------------------------------------------------------------
  // 6. Hinglish Technical & Conceptual Explanations
  // ---------------------------------------------------------------------------

  if (lower.includes("recursion samjha") || lower.includes("recursion samjhao") || lower.includes("recursion kya hota hai")) {
    return (
      "Dekh bhai, recursion simple shabdon me: jab ek function khud ko hi call kare jab tak koi base condition na hit ho jaye! " +
      "Jaise mirror ke samne mirror rakh do. " +
      "For example factorial:\n```cpp\nint fact(int n) {\n  if (n <= 1) return 1; // base condition\n  return n * fact(n - 1); // recursive step\n}\n```\n" +
      "Base condition zaruri hai warna infinite call stack crash ho jayega 😂 Samjha na?"
    );
  }

  if (lower.includes("ndvi simple language me") || lower.includes("ndvi simple language mein") || lower.includes("ndvi simple me")) {
    return (
      "Bhai simple bhasha me: NDVI se hum pata lagate hain ki khet ya ped-paudhe kitne hare-bhare aur healthy hain! " +
      "Healthy plants infrared light reflect karte hain aur red light absorb karte hain. " +
      "Formula simple hai: (NIR - Red) / (NIR + Red). " +
      "Agar score 0.6+ hai toh mast greenery hai, aur 0 ke paas hai toh banjar ya pani hai."
    );
  }

  if (lower.includes("ye kaise kaam karta hai") || lower.includes("yeh kaise kaam karta hai")) {
    return (
      "Basically iska flow kuch aisa hai: tum natural English ya query dalte ho, NexSpace usko samajhkar Sentinel-2 satellite data dhoondta hai STAC se, " +
      "aur phir AI models (jaise Grounding DINO aur multispectral engines) se scan karke exact coordinates aur evidence ke saath report deta hai!"
    );
  }

  if (lower.includes("what is remote sensing") && (lower.includes("bhai") || lower.includes("haan"))) {
    return (
      "Haan bhai 😎 Remote sensing basically dur se kisi object ya surface ki information gather karna hai bina physical contact ke—" +
      "jaise satellites (Sentinel-2) se dharti ki tasveerein lena aur unpe multispectral analysis chalana!"
    );
  }

  // ---------------------------------------------------------------------------
  // 7. General Knowledge Questions (Non-NexSpace)
  // ---------------------------------------------------------------------------

  if (lower.includes("capital of france")) {
    return "Paris 🇫🇷";
  }

  if (lower.includes("albert einstein") || lower.includes("einstein")) {
    return (
      "Albert Einstein was a German-born theoretical physicist widely acknowledged to be one of the greatest and most influential physicists of all time. " +
      "He is best known for developing the theory of relativity (E = mc²) and won the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect."
    );
  }

  if (lower === "what is ai?" || lower === "what is ai" || lower.includes("what is artificial intelligence")) {
    return (
      "Artificial Intelligence (AI) is the simulation of human intelligence by computer systems. " +
      "It encompasses machine learning, deep learning, computer vision, and natural language processing, allowing machines to recognize patterns, solve complex problems, and make data-driven decisions."
    );
  }

  if (lower === "what is python?" || lower === "what is python" || (lower.includes("python") && lower.includes("what is") && !lower.includes("nexspace"))) {
    return (
      "Python is a versatile, high-level programming language celebrated for its readable syntax and clean design. " +
      "It has a massive ecosystem spanning web development (Django, FastAPI), data analysis (pandas, NumPy), and AI/machine learning (PyTorch, TensorFlow)."
    );
  }

  if (lower.includes("what does nexspace use python for")) {
    return (
      "NexSpace uses Python for its dedicated FastAPI machine-learning backend microservice. " +
      "The Python service runs PyTorch model inference (Grounding DINO, BLIP), raster array operations (NumPy, Rasterio for NDVI/NDWI), and geospatial coordinate transforms."
    );
  }

  if (lower.includes("quantum computing")) {
    return (
      "Quantum computing utilizes the quantum mechanical phenomena of superposition and entanglement. " +
      "Instead of classical bits (0 or 1), quantum processors use qubits, enabling exponential computational parallelism for specific algorithms in cryptography, molecular modeling, and optimization."
    );
  }

  if (lower.includes("what should i learn today")) {
    return (
      "If you're into AI and geospatial tech, dive into STAC APIs and satellite spectral indices (NDVI/NDWI). " +
      "If you're building full-stack apps, explore Next.js Server Components and TypeScript utility types!"
    );
  }

  // ---------------------------------------------------------------------------
  // 8. General Technical & Coding Questions
  // ---------------------------------------------------------------------------

  if (lower.includes("binary search") && (lower.includes("c++") || lower.includes("cpp") || lower.includes("write"))) {
    return (
      "Here is an efficient Binary Search implementation in C++:\n\n" +
      "```cpp\n" +
      "#include <iostream>\n" +
      "#include <vector>\n\n" +
      "int binarySearch(const std::vector<int>& arr, int target) {\n" +
      "    int left = 0, right = arr.size() - 1;\n" +
      "    while (left <= right) {\n" +
      "        int mid = left + (right - left) / 2;\n" +
      "        if (arr[mid] == target) return mid;\n" +
      "        if (arr[mid] < target) left = mid + 1;\n" +
      "        else right = mid - 1;\n" +
      "    }\n" +
      "    return -1; // Element not found\n" +
      "}\n\n" +
      "int main() {\n" +
      "    std::vector<int> nums = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};\n" +
      "    int target = 23;\n" +
      "    int idx = binarySearch(nums, target);\n" +
      "    std::cout << \"Index of \" << target << \": \" << idx << std::endl;\n" +
      "    return 0;\n" +
      "}\n" +
      "```\n\n" +
      "Time complexity is O(log n) and space complexity is O(1)."
    );
  }

  if (lower.includes("recursion") && (lower.includes("explain") || lower.includes("what is"))) {
    return (
      "Recursion is a programming technique where a function solves a problem by calling a smaller instance of itself. " +
      "Every recursive function requires two core components:\n" +
      "1. Base Case: The terminating condition that stops recursion.\n" +
      "2. Recursive Step: The rule that reduces the problem closer to the base case.\n\n" +
      "Classic example (Factorial):\n```js\nfunction factorial(n) {\n  if (n <= 1) return 1; // Base case\n  return n * factorial(n - 1); // Recursive call\n}\n```"
    );
  }

  if (lower.includes("what is dns") || lower.includes("how does dns work") || lower.includes("what's dns")) {
    return (
      "DNS (Domain Name System) is the Internet's directory service. " +
      "It maps human-friendly domain names (e.g. `nexspace.ai`) to computer-readable numerical IP addresses (e.g. `142.250.190.46`), " +
      "allowing web browsers to route traffic accurately."
    );
  }

  if (lower.includes("http vs https") || lower.includes("difference between http and https")) {
    return (
      "The primary difference is encryption and security:\n" +
      "• HTTP (Hypertext Transfer Protocol): Transmits data in plain text over port 80. Vulnerable to eavesdropping and tampering.\n" +
      "• HTTPS (HTTP Secure): Uses TLS/SSL encryption over port 443 to encrypt all transmitted data, ensuring authentication and data integrity."
    );
  }

  if (lower.includes("what is an api") || lower.includes("what is api")) {
    return (
      "An API (Application Programming Interface) is a standardized set of protocols and rules that allows different software applications " +
      "to communicate and exchange data with each other—like how a frontend UI talks to a backend database."
    );
  }

  if (lower.includes("react vs next.js") || lower.includes("react vs next")) {
    return (
      "React is a client-side JavaScript library designed for rendering reactive component-based user interfaces. " +
      "Next.js is a full-stack production framework built on React that brings server-side rendering (SSR), static site generation (SSG), " +
      "React Server Components, automated route optimization, and built-in API handlers."
    );
  }

  // ---------------------------------------------------------------------------
  // 9. NexSpace-Specific Platform & Remote Sensing Questions
  // ---------------------------------------------------------------------------

  if (lower.includes("what is nexspace") || lower.includes("what's nexspace") || lower.includes("about nexspace")) {
    return (
      "NexSpace is a natural-language satellite and geospatial intelligence platform. " +
      "It lets users investigate satellite imagery using ordinary language and converts those queries into specialised remote-sensing analysis—" +
      "including open-vocabulary object detection, multispectral band math, temporal change detection, and Optical + SAR fusion."
    );
  }

  if (lower.includes("can nexspace calculate ndvi") || (lower.includes("calculate ndvi") && lower.includes("nexspace"))) {
    return (
      "Yes! NexSpace computes genuine NDVI (Normalized Difference Vegetation Index) directly from Sentinel-2 L2A satellite bands. " +
      "It uses the standard formula (NIR - Red) / (NIR + Red) via Band 8 (842 nm) and Band 4 (665 nm) at 10-meter ground resolution."
    );
  }

  if (lower.includes("ndvi") || lower.includes("vegetation index")) {
    return (
      "NDVI (Normalized Difference Vegetation Index) is a remote-sensing metric used to quantify vegetation health and density. " +
      "It is calculated as (NIR - Red) / (NIR + Red). Healthy green vegetation absorbs visible red light for photosynthesis and reflects near-infrared (NIR) light, " +
      "yielding high NDVI values (typically 0.4 to 0.85)."
    );
  }

  if (lower.includes("optical + sar") || lower.includes("optical sar") || lower.includes("how does optical + sar work")) {
    return (
      "NexSpace combines optical data (Sentinel-2) and Synthetic Aperture Radar (Sentinel-1 SAR). " +
      "SAR can penetrate clouds and darkness, while optical delivers rich multispectral color bands. " +
      "NexSpace co-registers both modalities into a shared CRS and fuses their features into a 1536-dimensional representation."
    );
  }

  if (lower.includes("how does it work") || lower.includes("how does nexspace work") || lower.includes("workflow")) {
    return (
      "The NexSpace investigation pipeline follows an automated 10-step process:\n" +
      "1. Natural-language Query Understanding (intent, targets, coordinates)\n" +
      "2. Input validation\n" +
      "3. Satellite Discovery via Element84 STAC\n" +
      "4. Specialist Model Routing\n" +
      "5. ML Execution (Grounding DINO, BLIP, spectral index engines)\n" +
      "6. Georeferencing in WGS84\n" +
      "7. Evidence extraction (bounding boxes, masks, metrics)\n" +
      "8. Provenance trace recording\n" +
      "9. Result synthesis and visual overlay generation\n" +
      "10. Explainable findings presentation"
    );
  }

  if (lower.includes("models") || lower.includes("grounding dino") || lower.includes("blip")) {
    return (
      "NexSpace utilizes specialized vision models:\n" +
      "• Grounding DINO: Open-vocabulary zero-shot object detection (vessels, infrastructure, aircraft)\n" +
      "• BLIP: Visual question answering (RSVQA) and descriptive scene captioning\n" +
      "• Spectral Index Engines: Direct NumPy/Rasterio band math for NDVI, NDWI, and NDBI\n" +
      "• Optical-SAR Fusion: CRS-aligned feature fusion layers"
    );
  }

  if (lower.includes("spatial reasoning")) {
    return (
      "NexSpace's spatial reasoning module evaluates geometric and topological relationships between detected entities—" +
      "such as 'along', 'closest_to', 'north_of', and distance offsets—anchoring natural-language spatial queries into real-world geographic coordinates."
    );
  }

  // ---------------------------------------------------------------------------
  // 10. Keyword Match Against FAQ Entries (Additive)
  // ---------------------------------------------------------------------------
  const words = lower.split(/\s+/).filter((w) => w.length > 3);
  let bestFaq: (typeof FAQ_ENTRIES)[number] | null = null;
  let bestScore = 0;

  for (const faq of FAQ_ENTRIES) {
    let score = 0;
    const targetText = `${faq.question} ${faq.answer} ${faq.tags.join(" ")}`.toLowerCase();
    for (const word of words) {
      if (targetText.includes(word)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestFaq = faq;
    }
  }

  if (bestFaq && bestScore >= 2) {
    return bestFaq.answer;
  }

  // ---------------------------------------------------------------------------
  // 11. General Conversational Fallback
  // ---------------------------------------------------------------------------
  return (
    "Hey! I'm NexSpace AI. You can ask me about satellite intelligence, NDVI, Grounding DINO detection, " +
    "coding, general science, or literally just have a casual chat! What are you working on today?"
  );
}
