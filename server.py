from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from io import BytesIO

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = r'/home/gunicornuser/project/Plant_disease/Model'
model = ViTForImageClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Function to predict plant disease from image bytes
def predict_plant_disease(image_bytes):
    image = Image.open(BytesIO(image_bytes))    
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    
    # Get the predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    confidence = probabilities[0][predicted_class].item() * 100  # Convert to percentage
     # Format confidence as a percentage string
    confidence_str = f"{confidence:.2f}%"  # Round to 2 decimal places
    return predicted_class.item(), confidence_str


# Disease information
def disease_information(predicted_class):
    diseases = {
        0: {
            'name': 'Alternaria Leaf Spot',
            'description': 'Alternaria leaf spot is characterized by small, dark, circular spots with concentric rings (target-like spots) on leaves, which can cause leaf yellowing and drop. In severe cases, the spots can coalesce, leading to large dead areas on the leaf.',
            'conditions': 'Warm and humid conditions favor the growth and spread of Alternaria fungi. Poor air circulation can exacerbate the problem, as it keeps the foliage wet for extended periods. High moisture levels, including heavy dew, rainfall, or overhead irrigation, can promote fungal growth.',
            'mechanical_treatment': 'Remove and destroy infected plant debris to reduce sources of inoculum. Ensure good air circulation around plants by proper spacing and pruning. Water plants at the base to avoid wetting foliage and reduce leaf wetness duration.',
            'chemical_treatment': 'Chlorothalonil:Usage: Apply every 7-10 days during conditions favorable for disease development.Application: Follow label instructions for specific crops and timing.Effectiveness: Controls a broad spectrum of fungal pathogens, including Alternaria.Copper-Based Fungicides (e.g., Copper Hydroxide, Copper Oxychloride):Usage: Apply at the first sign of disease and repeat applications every 7-10 days as necessary.Application: Use according to the manufacturer\'s instructions.Effectiveness: Provides protection against many fungal and bacterial diseases.Mancozeb:Usage: Apply as a preventative treatment or at the first sign of symptoms.Application: Follow label directions for dosage and application frequency.Effectiveness: Effective against a variety of fungal pathogens, including Alternaria.Azoxystrobin:Usage: Apply as a foliar spray, typically every 7-14 days depending on disease pressure.Application: Use according to the label instructions for specific crops.Effectiveness: A systemic fungicide that provides both protective and curative action against Alternaria.Difenoconazole:Usage: Apply as a foliar spray at the first sign of disease.Application: Follow the label for specific crops and timing.Effectiveness: Effective in controlling Alternaria and other fungal pathogens.','source':'https://en.wikipedia.org/wiki/Alternaria_leaf_spot'
        },
        1: {
            'name': 'Anthracnose',
            'description': 'Anthracnose is a fungal disease that tends to attack plants in the spring when the weather is cool and wet, primarily on leaves and twigs. The fungi overwinter in dead twigs and fallen leaves. Cool, rainy weather creates perfect conditions for the spores to spread. Dry and hot weather stop the progression of the disease that may begin again once the weather conditions become optimal. The problem can be cyclic but is rarely fatal. Anthracnose fungus infects many deciduous and evergreen trees and shrubs, as well as fruits, vegetables, and grass. Anthracnose is noticeable along the leaves and the veins as small lesions. These dark, sunken lesions may also be found on stems, flowers, and fruits. In order to distinguish between anthracnose and other leaf spot diseases, you should carefully examine the undersides of leaves for a number of small tan to brown dots, about the size of a pinhead.',
            'conditions': 'Anthracnose symptoms vary by plant host and due to weather conditions. On landscape trees, the fungi infect developing shoots and expanding leaves. Small beige, brown, black, or black spots later appear on infected twigs of hosts such as elm, oak, and sycamore.',
            'mechanical_treatment': 'Anthracnose control begins with practicing good sanitation. Picking up and disposing of all diseased plant parts, including twigs and leaves, from the ground or from around the plant, is important. This keeps the fungus from overwintering near the plant. Proper pruning techniques to rid trees and plants of old and dead wood also helps with the prevention of anthracnose fungus. Keeping plants healthy by providing proper light, water, and fertilizer will strengthen the plant\'s ability to ward off a fungus attack. Stressed trees and plants have a difficult time recovering from anthracnose fungus. Chemical treatment is rarely used except when the disease involves newly transplanted plants or continual defoliation.',
            'chemical_treatment': 'Chlorothalonil, Mancozeb, Copper-based fungicides.Chlorothalonil: Effective against a broad spectrum of fungal diseases, including Anthracnose. Commonly used on vegetables, fruits, and ornamentals. Apply according to label instructions, typically every 7-14 days during the growing season. Mancozeb: A broad-spectrum fungicide that is effective against Anthracnose. Often used on fruits, vegetables, and ornamental plants. Typically applied every 7-10 days, particularly during wet conditions when the disease is more prevalent. Copper-based fungicides: Effective in controlling Anthracnose on a variety of plants, including vegetables, fruits, and ornamental trees. Examples include Copper Hydroxide and Copper Oxychloride. Apply at the first sign of disease and repeat applications every 7-10 days as necessary.',
            'source':'https://www.maine.gov/dacf/php/gotpests/diseases/anthracnose.htm#:~:text=Anthracnose%20is%20a%20group%20of,attacks%20only%20specific%20tree%20species.'
        },
        2: {
            'name': 'Black Spot leaf disease',
            'description': 'Black spot is a fungal disease that primarily affects plants with fleshy leaves and stems. The disease manifests as round, black spots with fringed margins on the upper sides of leaves. These spots may enlarge and merge, causing significant damage. Infected leaves often turn yellow and drop prematurely, weakening the plant over time. Black spot thrives in warm, humid conditions and spreads rapidly as temperatures rise into the 70s (Fahrenheit).',
            'conditions': 'Cause: Black spot is a fungal disease that affects plants with fleshy leaves and stems. It thrives in hot, dry conditions, and spreads rapidly as temperatures rise into the 70s.',
            'mechanical_treatment': 'Remove diseased canes and leaves, keep foliage dry, and consider resistant rose varieties or fungicide application.',
            'chemical_treatment': 'Fungicides, neem oil, and baking soda are effective treatments for black spot control. Fungicides are labeled for black spot control, while neem oil has antifungal properties. Baking soda and horticultural oil can be mixed to alter leaf surface pH. Apply treatments early and consistently to prevent further spread.',
            'source':'https://en.wikipedia.org/wiki/Black_spot_leaf_disease'
        },
        3: {
            'name': 'Botrytis Blight',
            'description': 'Botrytis blight, also known as gray mold, causes gray, fuzzy mold on flowers, leaves, and stems. Infected plant parts may become soft, brown, and decayed, often leading to the death of the affected tissues.',
            'conditions': 'Cool, damp conditions are ideal for the development of Botrytis cinerea. Overcrowded plants reduce air circulation, increasing humidity and moisture retention around plants. High humidity levels, especially in greenhouses or during prolonged periods of wet weather, can exacerbate the disease.',
            'mechanical_treatment': 'Remove affected plant parts and dispose of them to reduce sources of infection. Increase air circulation around plants by proper spacing and pruning. Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.',
            'chemical_treatment': 'Fungicides with fenhexamid, iprodione, or chlorothalonil can be effective.Elevate (fenhexamid): Specifically targets Botrytis and provides protective and curative action. Chipco 26019 (iprodione): A fungicide effective against Botrytis and other fungal diseases. Daconil (chlorothalonil): Provides broad-spectrum control against many fungal pathogens.',
            'source' : 'https://plantpathology.ca.uky.edu/files/ppfs-gen-19.pdf'
        },
        4: {
            'name': 'Chlorosis',
            'description': 'Chlorosis is a condition where leaves produce insufficient chlorophyll, turning yellow while the veins remain green. It can affect entire plants or individual leaves and is often a symptom of underlying issues.',
            'conditions': 'Nutrient deficiencies, particularly iron, manganese, or nitrogen, can cause chlorosis. Poor soil drainage can lead to root damage and impaired nutrient uptake. Root damage due to physical injury or disease can also cause chlorosis.',
            'mechanical_treatment': 'Correct nutrient deficiencies by using appropriate fertilizers tailored to the specific nutrient needs. Improve soil drainage to prevent waterlogging and root damage. Avoid damaging roots during cultivation and provide proper plant care to enhance root health.',
            'chemical_treatment': 'Fertilizers containing the deficient nutrient can correct the specific deficiency causing chlorosis.Iron chelates (e.g., Iron EDTA): For correcting iron deficiency, which is a common cause of chlorosis. Manganese sulfate: Used to correct manganese deficiency, another cause of chlorosis. Nitrogen-rich fertilizers: For correcting nitrogen deficiency, which can lead to overall yellowing of leaves.','source' : 'https://en.wikipedia.org/wiki/Chlorosis'
        },
        5: {
            'name': 'Curl Leaf',
            'description': 'Curl leaf is a condition where leaves curl and distort. It can be caused by a variety of factors, including pests, diseases, and environmental stress.',
            'conditions': 'Pest infestations, such as aphids, mites, and thrips, can cause leaf curling. Viral infections can also lead to leaf curl symptoms. Environmental stress, such as drought, excessive watering, or nutrient imbalances, can result in leaf curling.',
            'mechanical_treatment': 'Remove and destroy affected plant parts to reduce sources of pests and diseases. Implement proper pest control measures to manage infestations. Maintain optimal growing conditions, including proper watering and fertilization.',
            'chemical_treatment': 'Pesticides, fungicides, and appropriate fertilization can help manage the underlying causes of curl leaf.Insecticidal soap or neem oil: For controlling soft-bodied pests like aphids and mites. Fungicides: For managing fungal infections that cause leaf curl. Balanced fertilizers: To address nutrient imbalances that can lead to leaf curling.','source' : 'https://en.wikipedia.org/wiki/Leaf_curl'
        },
        6: {
            'name': 'Die black',
            'Description': 'Dieback is a condition where the branches and leaves of a plant die from the tips inward, leading to the death of the affected parts. This condition can affect a wide range of plants, including trees, shrubs, and perennials. The causes of dieback can vary and include fungal infections, bacterial diseases, pest infestations, environmental stress, and physical damage.','conditions':'Fungal Infections: Common fungal pathogens causing dieback include Botryosphaeria, Phytophthora, and Verticillium species. These fungi can infect through wounds or natural openings and cause the death of plant tissues.Bacterial Diseases: Bacterial pathogens such as Pseudomonas and Xanthomonas can also lead to dieback by infecting plant tissues and causing necrosis.Pest Infestations: Insects like borers and bark beetles can damage plant tissues, leading to dieback.Environmental Stress: Drought, excessive water, nutrient deficiencies, and extreme temperatures can weaken plants, making them more susceptible to dieback.Physical Damage: Mechanical injuries from pruning, storm damage, or human activity can provide entry points for pathogens, leading to dieback.','mechanical_treatment' : 'Sanitation:Remove and destroy infected or dead plant parts to prevent the spread of pathogens.Prune back affected branches to healthy tissue, making clean cuts to minimize damage.Watering:Ensure plants receive adequate and consistent water, avoiding both drought stress and waterlogging.Water at the base of plants to keep foliage dry and reduce the risk of infection.Soil Management:Improve soil drainage to prevent waterlogging, which can promote root infections.Mulch around the base of plants to retain moisture and regulate soil temperature.Nutrient Management:Provide balanced fertilization to maintain plant health and resistance to stress and infection.Avoid over-fertilization, which can lead to excessive growth and susceptibility to dieback.','chemical_treatment' : 'Fungal Dieback:Use fungicides like chlorothalonil, copper-based fungicides, or mancozeb as a preventative measure or at the first sign of infection.Reapply treatments as per the manufacturer\'s instructions, usually every 7-10 days, especially in humid conditions.Bacterial Dieback:Copper-based bactericides can help control bacterial pathogens. Apply according to label directions and reapply as needed.Insect-Related Dieback:Use appropriate insecticides to manage pest infestations. Target specific pests like borers or bark beetles, following the manufacturer\'s recommendations for application.','source' : 'https://www.livelyroot.com/blogs/plant-care/what-causes-black-leaves'
        },
        7: {
            'name': 'Downy Mildew',
            'description': 'Downy mildew is a fungal disease that causes yellow or white patches on the upper leaf surfaces, with corresponding fuzzy growth on the undersides. Infected leaves may curl, distort, and drop prematurely.',
            'conditions': 'Cool, moist conditions favor the development of downy mildew. Extended periods of leaf wetness, high humidity, and moderate temperatures (50-70°F) promote fungal growth and infection.',
            'mechanical_treatment': 'Remove and destroy infected plant debris to reduce sources of inoculum. Improve air circulation around plants by proper spacing and pruning. Water plants early in the day to allow foliage to dry before nightfall.',
            'chemical_treatment': 'Fungicides with metalaxyl, fosetyl-Al, or copper-based compounds can be effective.Ridomil Gold (metalaxyl): Provides systemic control of downy mildew and other oomycete pathogens. Aliette (fosetyl-Al): A systemic fungicide effective against downy mildew. Copper fungicides (e.g., Copper Oxychloride): Effective against downy mildew when applied preventatively.' ,'source' : 'https://en.wikipedia.org/wiki/Downy_mildew'
        },
        8 :{
            'name': 'Healthy',
            'decription' : 'This is not a disease this is a healthy plant leaf',
            'conditions' : 'Taking good care of your soild and plant',
            'mechaincal_treatment' : 'This is not a disease to have a treatment',
            'chemical_treatment' : 'This is not a disease to have a treatment',
            'source' : 'https://en.wikipedia.org/wiki/Leaf'
        },
        
        9:{
             'name': 'Leaf Blight',
            'description': 'Leaf blight is a general term for diseases that cause browning and death of leaf tissue. It can be caused by fungi, bacteria, or environmental factors.',
            'conditions': 'Wet and humid conditions favor the development and spread of leaf blight pathogens.Fungal or bacterial pathogens are often the primary cause of leaf blight.Poor air circulation and overcrowding can exacerbate the problem.',
            'mechanical_treatment': 'Remove and destroy affected leaves to reduce sources of inoculum.Improve air circulation around plants by proper spacing and pruning.Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.',
            'chemical_treatment': 'Fungicides containing chlorothalonil, mancozeb, or copper-based compounds can be effective against fungal blight.Bactericides may be needed if bacterial blight is present.Daconil (chlorothalonil): Provides broad-spectrum control against many fungal pathogens.Mancozeb: Effective against a wide range of fungal diseases.Copper fungicides (e.g., Bordeaux mixture): Effective against many fungal and bacterial diseases.Streptomycin: An antibiotic effective against bacterial blight.','source' : 'https://en.wikipedia.org/wiki/Blight'
        },
        13: {
            'name': 'Powdery Mildew',
            'description': 'Powdery mildew is a fungal disease characterized by white, powdery fungal growth on the upper and lower leaf surfaces, stems, flowers, and fruits. Infected plant parts may become distorted and stunted.',
            'conditions': 'Warm, dry conditions with high humidity (but not leaf wetness) favor the development of powdery mildew. Overcrowded plants with poor air circulation are more susceptible to the disease.',
            'mechanical_treatment': 'Remove and destroy infected plant parts to reduce sources of inoculum. Improve air circulation around plants by proper spacing and pruning. Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.',
            'chemical_treatment': 'Fungicides with sulfur, potassium bicarbonate, or myclobutanil can be effective.Sulfur: Provides effective control of powdery mildew when applied preventatively. Kaligreen (potassium bicarbonate): A contact fungicide that disrupts fungal cell walls. Rally (myclobutanil): A systemic fungicide that provides curative and protective action against powdery mildew.','source' : 'https://en.wikipedia.org/wiki/Powdery_mildew'
        },
        14: {
            'name': 'Rust',
            'description': 'Rust is a fungal disease characterized by orange, yellow, or brown pustules on the undersides of leaves. Infected leaves may become distorted, yellow, and drop prematurely. Severe infections can weaken the plant over time.',
            'conditions': 'Moderate temperatures (50-75°F) and high humidity favor the development of rust. Extended periods of leaf wetness and overcrowded plants with poor air circulation increase the risk of infection.',
            'mechanical_treatment': 'Remove and destroy infected plant debris to reduce sources of inoculum. Improve air circulation around plants by proper spacing and pruning. Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.',
            'chemical_treatment': 'Fungicides with myclobutanil, azoxystrobin, or sulfur can be effective.Rally (myclobutanil): A systemic fungicide that provides curative and protective action against rust. Heritage (azoxystrobin): A broad-spectrum fungicide that controls rust and other fungal diseases. Sulfur: Provides effective control of rust when applied preventatively.','source' : 'https://en.wikipedia.org/wiki/Rust_(fungus)'
        },
        10:{
            'name': 'Leaf Necrosis',
            'description': 'Leaf necrosis refers to the death of leaf tissue, often presenting as brown or black spots or patches. It can result from fungal or bacterial infections, chemical damage, or environmental stress.',
            'conditions' : 'Fungal or bacterial infections can cause necrotic spots or patches on leaves. Chemical damage from herbicides, pesticides, or fertilizers can lead to necrosis.Environmental stress, such as drought, extreme temperatures, or nutrient imbalances, can cause leaf tissue to die.',
            'mechanical_treatment' : 'Remove and destroy affected leaves to reduce sources of inoculum.Improve air circulation around plants by proper spacing and pruning.Avoid overhead watering and water early in the day to allow foliage to dry before nightfall',
            'chemical_treatment' : 'Fungicides containing chlorothalonil, mancozeb, or copper-based.compounds can be effective against fungal blight.Bactericides may be needed if bacterial blight is present.Daconil (chlorothalonil): Provides broad-spectrum control against many fungal pathogens.Mancozeb: Effective against a wide range of fungal diseases.Copper fungicides(e.g., Bordeaux mixture): Effective against many fungal and bacterial diseases.Streptomycin: An antibiotic effective against bacterial blight.','source' : 'https://hortsense.cahnrs.wsu.edu/fact-sheet/common-cultural-marginal-leaf-necrosis/#:~:text=Biology,excessive%20heat%2C%20and%20chemical%20injury.'
        },
        11:{
             'name': 'Leaf Spot',
            'description': 'Leaf spot is a general term used to describe a variety of diseases caused by different pathogens such as fungi, bacteria, and sometimes viruses. The disease is characterized by spots on the leaves that can vary in size, shape, and color depending on the causative agent. The spots are typically round and can be brown, black, or yellow with a darker margin. Severe infections can lead to premature leaf drop, reducing the plant\'s vigor and yield.',
            'conditions' : 'Humidity and Moisture: High humidity and prolonged leaf wetness create an ideal environment for the development and spread of leaf spot pathogens.Temperature: Moderate temperatures (15-30°C) are typically conducive to leaf spot infections, though this can vary depending on the specific pathogen.Crowded Plantings: Dense foliage and poor air circulation can trap moisture and increase the risk of infection.Infected Plant Debris: Pathogens can survive in infected plant debris and soil, facilitating the spread of the disease.',
            'mechanical_treatment' : 'Sanitation: Remove and destroy infected leaves and plant debris to reduce sources of inoculum.Watering: Water plants early in the day to allow leaves to dry quickly, and avoid overhead irrigation.Spacing: Ensure adequate spacing between plants to improve air circulation.Resistant Varieties: Plant disease-resistant varieties when available.',
            'chemical_treatment' : 'Copper-Based Fungicides: Effective against many types of leaf spot diseases. Apply according to the manufacturer\'s instructions, usually before symptoms appear or at the first sign of disease.Chlorothalonil: A broad-spectrum fungicide that can control leaf spot diseases. Follow label directions for application rates and timing.Mancozeb: Another fungicide that is effective against leaf spot diseases. Apply as a preventative treatment or when symptoms first appear.Bacterial Leaf Spot: For bacterial infections, copper-based fungicides can be effective. Additionally, some antibiotics like streptomycin are sometimes used, though they are less common.Fungal Leaf Spot: Common fungicides like mancozeb, chlorothalonil, and thiophanate-methyl can be effective. Regular applications during the growing season may be necessary, particularly in humid or wet conditions.','source' : 'https://en.wikipedia.org/wiki/Leaf_spot'
        },
        12:{
            'name': 'Mosaic Virus',
            'description': ' any virus that causes infected plant foliage to have a mottled appearance. Such viruses come from a variety of unrelated lineages and consequently there is no taxon that unites all mosaic viruses.',
            'conditions' : ' Tomato mosaic virus and tobacco mosaic virus.   Mosaic viruses are plant viruses that cause mottling, discoloration, and distortion of leaves in infected plants. They can be transmitted by aphids or contaminated seeds or cuttings, affecting various crops like roses, beans, tomatoes, potatoes, cucumbers, pumpkins, squash, melons, and peppers. Symptoms include yellow, white, or green stripes, wrinkled leaves, yellowing veins, stunted growth, mottled fruit, and dark green blisters on stems.',
            'mechanical_treatment' : 'No cure, but prevention involves removing infected plants and avoiding spread.',
            'chemical_treatment' : 'Mosaic viruses, which affect crops like tomatoes, cucumbers, and peppers, are uncurable due to lack of chemical treatments. Preventive measures include maintaining garden hygiene, promptly removing infected plants, and controlling insect vectors like aphids to reduce transmission.Since there is a lot of virsues from this family we can\'t recommend a specific fungicide until you specify the virus','source' : 'https://en.wikipedia.org/wiki/Mosaic_virus'
        },
       15:{
           'name': 'Septoria Leaf Spot',
            'description': ' Septoria leaf spot is a fungal disease caused by the pathogen Septoria lycopersici, commonly affecting tomato plants. It is characterized by small, water-soaked spots that turn brown and are surrounded by a yellow halo. The spots eventually coalesce, causing significant damage to the foliage. Severe infections can lead to defoliation, reducing the plant’s ability to photosynthesize and thus diminishing fruit yield and quality.',
            'conditions' : ' Humidity and Moisture: High humidity and prolonged periods of leaf wetness favor the development and spread of Septoria leaf spot.Temperature: Moderate temperatures (20-25°C) are optimal for the growth and infection of Septoria lycopersici.Plant Debris: The pathogen overwinters in infected plant debris and soil, facilitating infection during the growing season.Overhead Irrigation: Splashing water can spread the spores from infected to healthy leaves.',
            'mechanical_treatment' : 'Sanitation: Remove and destroy infected leaves and plant debris to reduce sources of inoculum.Watering: Water plants at the base to keep foliage dry and avoid overhead irrigation.Crop Rotation: Rotate crops with non-host plants to reduce soilborne inoculum.Resistant Varieties: Plant disease-resistant tomato varieties when available.',
            'chemical_treatment' : 'Chlorothalonil: A broad-spectrum fungicide effective against Septoria leaf spot. Apply according to the manufacturer\'s instructions, typically every 7-10 days during favorable conditions for the disease.Copper-Based Fungicides: Effective against a range of fungal diseases, including Septoria leaf spot. Follow label directions for application rates and timing.Mancozeb: Another fungicide effective in controlling Septoria leaf spot. Regular applications during the growing season may be necessary.Fungicide Sprays: Use fungicides such as chlorothalonil, copper-based fungicides, or mancozeb as a preventative measure or at the first sign of disease. Reapply as per the manufacturer\'s instructions, usually every 7-10 days, especially in humid conditions.Rotation of Fungicides: To prevent the development of resistance, rotate between different fungicides with different modes of action.', 'source' : 'https://extension.wvu.edu/lawn-gardening-pests/plant-disease/fruit-vegetable-diseases/septoria-leaf-spot'
       },
       16:{
           'name': 'Sooty mold',
            'description': ' Sooty mold is a fungal disease that grows on plants and other surfaces covered by honeydew, a sticky substance created by certain insects. Sooty mold\'s name comes from the dark threadlike growth (mycelium) of the fungi resembling a layer of soot.',
            'conditions' : ' Sooty mold is a fungal disease that forms a black coating on plant\'s leaves and stems, blocking sunlight and reducing photosynthesis. It can also diminish a plant\'s aesthetic value.',
            'mechanical_treatment' : 'To combat sooty mold, identify the source, eliminate the pest, wash off the mold, and consider using neem oil as a treatment. Although sooty mold isn\'t lethal, immediate action is crucial to prevent further damage.',
            'chemical_treatment' : 'Chemical treatment for sooty mold involves washing off mold with a hosepipe, controlling insect vectors with contact insecticides, and preventing infestations by regularly cleaning foliage and managing pests.Horticultural Oil: This can be effective in suffocating the mold by coating the surface where the mold is growing.Neem Oil: Known for its antifungal properties, neem oil can help control sooty mold when applied to affected plants.Copper Fungicides: These are effective against a wide range of fungal diseases, including sooty mold. However, they should be used cautiously as they can be harsh on some plants.Sulfur-Based Fungicides: These can also be used to control sooty mold, but like copper fungicides, they should be used carefully to avoid damage to sensitive plants.Systemic Fungicides: These are absorbed by the plant and can provide longer-term control of sooty mold. They are typically used when the mold is persistent and widespread.','source' : 'https://en.wikipedia.org/wiki/Sooty_mold'
       },
       17:{
            'name': 'Sooty Mould',
            'description': ' Sooty mould is a fungal disease characterized by the presence of black, soot-like deposits on the surface of leaves, stems, and fruits of plants. It is not a parasitic fungus; instead, it grows on the honeydew excreted by sap-sucking insects such as aphids, whiteflies, scale insects, and mealybugs. The black coating can interfere with photosynthesis by blocking sunlight, leading to reduced plant vigor and growth.',
            'conditions' : ' Sooty mould develops on the honeydew secretions from sap-sucking insects. These insects feed on plant sap and excrete a sugary substance called honeydew, which creates a favorable environment for the growth of sooty mould fungi. The primary factors contributing to sooty mould infection include:Presence of Sap-Sucking Insects: Aphids, whiteflies, scale insects, and mealybugs are common culprits.High Humidity: Moist conditions favor the growth of sooty mould fungi.Poor Air Circulation: Dense foliage and overcrowded plantings can contribute to an increase in humidity and moisture retention.',
            'mechanical_treatment' : 'Pruning: Regularly prune and thin out plants to improve air circulation and reduce humidity.Washing: Wash affected plants with a strong stream of water to remove both the honeydew and the sooty mould.Insect Control: Manage and control the population of sap-sucking insects using natural predators or insecticidal soaps to prevent honeydew formation.Cleaning: Remove and destroy heavily infested plant parts to reduce sources of honeydew and sooty mould.',
            'chemical_treatment' : 'Insecticidal Soaps: Effective in controlling sap-sucking insects that produce honeydew. Apply according to label instructions, ensuring thorough coverage of the plant.Horticultural Oils: These oils can suffocate insects and reduce honeydew production. Apply during cooler parts of the day to avoid leaf burn.Systemic Insecticides: These are absorbed by the plant and can provide longer-lasting protection against sap-sucking insects. Follow label directions for proper use.Neem Oil:A natural oil that can control a wide range of sap-sucking insects.Apply every 7-14 days, depending on the severity of the infestation.Imidacloprid:A systemic insecticide that is effective against sap-sucking insects.Apply as a soil drench or foliar spray according to label instructions.Pyrethrin-Based Insecticides:Effective for quick knockdown of insects like aphids and whiteflies.Use as a contact spray and reapply as needed based on label instructions.Horticultural Oil (Petroleum-based or Plant-based):Controls insects by suffocation and can reduce sooty mould by eliminating honeydew production.Apply during dormant season for best results, and follow label directions carefully to avoid plant damage.','source' : 'https://www.rhs.org.uk/biodiversity/sooty-moulds'
       }
       
    }
    
    return diseases.get(predicted_class, {'error': 'Disease information not available'})

# Initialize Flask app
app = Flask(__name__)
 
# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    predicted_class, confidence = predict_plant_disease(image_bytes)
    disease_info = disease_information(predicted_class)
    
    response = {
        'disease': disease_info,
        'confidence': confidence
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
