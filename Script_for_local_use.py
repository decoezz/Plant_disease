import torch
import os
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = r'C:\Users\decoe\Desktop\Comptition\Model'
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

# Function to predict plant disease
def predict_plant_disease(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    
    # Get the predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)
    
    return predicted_class.item(), probabilities[0][predicted_class].item()

# Get image path from the user
image_path = input("Enter the path to the plant image: ")

# Test the function
predicted_class, confidence = predict_plant_disease(image_path)
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.4f}')

if predicted_class == 0: 
    print('Alternaria Leaf Spot')
    print('''
Description: 
          Alternaria leaf spot is characterized by small, dark, circular spots with concentric rings (target-like spots) on leaves, which can cause leaf yellowing and drop. In severe cases, the spots can coalesce, leading to large dead areas on the leaf.
Conditions of Infection:
          Warm and humid conditions favor the growth and spread of Alternaria fungi.
          Poor air circulation can exacerbate the problem, as it keeps the foliage wet for extended periods.
          High moisture levels, including heavy dew, rainfall, or overhead irrigation, can promote fungal growth.
Mechanical Treatment:
          Remove and destroy infected plant debris to reduce sources of inoculum.
          Ensure good air circulation around plants by proper spacing and pruning.
          Water plants at the base to avoid wetting foliage and reduce leaf wetness duration.
Chemical Treatment:
          Fungicides containing chlorothalonil, mancozeb, or copper-based compounds can be effective.
Specific Chemical Treatments:
          Daconil (chlorothalonil): A broad-spectrum fungicide that can control Alternaria and other fungal diseases.
          Mancozeb: A fungicide that provides protective action against a wide range of fungal diseases.
          Copper fungicides (e.g., Bordeaux mixture): Effective against many fungal and bacterial diseases; apply according to label instructions.
    ''')
elif  predicted_class == 1:
    print('Anthracnose')
    print('''
Description:
          Anthracnose is a fungal disease that tends to attack plants in the spring when the weather is cool and wet, primarily on leaves and twigs. The fungi overwinter in dead twigs and fallen leaves. Cool, rainy weather creates perfect conditions for the spores to spread. Dry and hot weather stop the progression of the disease that may begin again once the weather conditions become optimal. The problem can be cyclic but is rarely fatal. Anthracnose fungus infects many deciduous and evergreen trees and shrubs, as well as fruits, vegetables, and grass. Anthracnose is noticeable along the leaves and the veins as small lesions. These dark, sunken lesions may also be found on stems, flowers, and fruits. In order to distinguish between anthracnose and other leaf spot diseases, you should carefully examine the undersides of leaves for a number of small tan to brown dots, about the size of a pinhead.
Conditions of Infection:
          Anthracnose symptoms vary by plant host and due to weather conditions. On landscape trees, the fungi infect developing shoots and expanding leaves. Small beige, brown, black, or black spots later appear on infected twigs of hosts such as elm, oak, and sycamore.
Mechanical Treatment:
          Anthracnose control begins with practicing good sanitation. Picking up and disposing of all diseased plant parts, including twigs and leaves, from the ground or from around the plant, is important. This keeps the fungus from overwintering near the plant. Proper pruning techniques to rid trees and plants of old and dead wood also helps with the prevention of anthracnose fungus. Keeping plants healthy by providing proper light, water, and fertilizer will strengthen the plant's ability to ward off a fungus attack. Stressed trees and plants have a difficult time recovering from anthracnose fungus. Chemical treatment is rarely used except when the disease involves newly transplanted plants or continual defoliation.
Chemical Treatment:
          Chlorothalonil:
          Effective against a broad spectrum of fungal diseases, including Anthracnose.
          Commonly used on vegetables, fruits, and ornamentals.
          Apply according to label instructions, typically every 7-14 days during the growing season​.
          Mancozeb:
          A broad-spectrum fungicide that is effective against Anthracnose.
          Often used on fruits, vegetables, and ornamental plants.
          Typically applied every 7-10 days, particularly during wet conditions when the disease is more prevalent​.
          Copper-based fungicides:
          Effective in controlling Anthracnose on a variety of plants, including vegetables, fruits, and ornamental trees.
          Examples include Copper Hydroxide and Copper Oxychloride.
          Apply at the first sign of disease and repeat applications every 7-10 days as necessary​.

''')

elif  predicted_class == 2:
    print('Black Spot')
    print('''
Description: 
          Black spot is a fungal disease that primarily affects plants with fleshy leaves and stems. The disease manifests as round, black spots with fringed margins on the upper sides of leaves. These spots may enlarge and merge, causing significant damage. Infected leaves often turn yellow and drop prematurely, weakening the plant over time. Black spot thrives in warm, humid conditions and spreads rapidly as temperatures rise into the 70s (Fahrenheit).
conditions of infection:
          Cause:  Black spot is a fungal disease that affects plants with fleshy leaves and stems. It thrives in hot, dry conditions, and spreads rapidly as temperatures rise into the 70s. Treatment options include fungicides, neem oil, or a mixture of baking soda and horticultural oil. Regular inspection and early detection are crucial to prevent further spread. Neem oil is a natural fungicide, while baking soda can be used as a natural solution.
Mechanical Treatment: 
          Remove diseased canes and leaves, keep foliage dry, and consider resistant rose varieties or fungicide application.
Chemical Treatment: 
          Fungicides, neem oil, and baking soda are effective treatments for black spot control. Fungicides are labeled for black spot control, while neem oil has antifungal properties. Baking soda and horticultural oil can be mixed to alter leaf surface pH. Apply treatments early and consistently to prevent further spread.
''')
elif predicted_class == 3:
    print('Botrytis Blight')
    print('''
Description: 
          Botrytis blight, also known as gray mold, causes gray, fuzzy mold on flowers, leaves, and stems. Infected plant parts may become soft, brown, and decayed, often leading to the death of the affected tissues.
Conditions of Infection:
          Cool, damp conditions are ideal for the development of Botrytis cinerea.
          Overcrowded plants reduce air circulation, increasing humidity and moisture retention around plants.
          High humidity levels, especially in greenhouses or during prolonged periods of wet weather, can exacerbate the disease.
Mechanical Treatment:
          Remove affected plant parts and dispose of them to reduce sources of infection.
          Increase air circulation around plants by proper spacing and pruning.
          Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.
Chemical Treatment:
          Fungicides with fenhexamid, iprodione, or chlorothalonil can be effective.
Specific Chemical Treatments:
          Elevate (fenhexamid): Specifically targets Botrytis and provides protective and curative action.
          Chipco 26019 (iprodione): A fungicide effective against Botrytis and other fungal diseases.
          Daconil (chlorothalonil): Provides broad-spectrum control against many fungal pathogens.
''')
elif predicted_class == 4:
    print('Chlorosis')
    print('''
Description: 
          Chlorosis is a condition where leaves produce insufficient chlorophyll, turning yellow while the veins remain green. It can affect entire plants or individual leaves and is often a symptom of underlying issues.
Conditions of Infection:
          Nutrient deficiencies, particularly iron, manganese, or nitrogen, can cause chlorosis.
          Poor soil drainage can lead to root damage and impaired nutrient uptake.
          Root damage due to physical injury or disease can also cause chlorosis.
Mechhanical Treatment:
          Correct nutrient deficiencies by using appropriate fertilizers tailored to the specific nutrient needs.
          Improve soil drainage to prevent waterlogging and root damage.
          Avoid damaging roots during cultivation and provide proper plant care to enhance root health.
Chemical Treatment:
          Fertilizers containing the deficient nutrient can correct the specific deficiency causing chlorosis.
Specific Chemical Treatments:
          Iron chelates (e.g., Iron EDTA): For correcting iron deficiency, which is a common cause of chlorosis.
          Manganese sulfate: Used to correct manganese deficiency, another cause of chlorosis.
          Nitrogen-rich fertilizers: For correcting nitrogen deficiency, which can lead to overall yellowing of leaves.
''')
elif predicted_class ==5:
    print('Curl Leaf')
    print('''
Description: 
          Curl leaf is a condition where leaves curl and distort. It can be caused by a variety of factors, including pests, diseases, and environmental stress.
Conditions of Infection:
          Pest infestations, such as aphids, mites, and thrips, can cause leaf curling.
          Viral infections can also lead to leaf distortion and curling.
          Environmental stress, including drought, excessive watering, or temperature extremes, can cause leaves to curl.
Mechanical Treatment:
          Control pests using insecticidal soaps, neem oil, or specific insecticides.
          Remove and destroy infected plants if a viral infection is suspected.
          Adjust watering practices to avoid stress, ensuring plants receive adequate but not excessive water.
Chemical Treatment:
          Insecticides or miticides for controlling pests that cause leaf curling.
Specific Chemical Treatments:
          Neem oil: A natural insecticide effective against a wide range of pests.
          Insecticidal soap: Safe for use on most plants and effective against soft-bodied pests like aphids and mites.
          Pyrethrin-based insecticides: Provide quick knockdown of pests and are effective against a variety of insects.
''')

elif predicted_class == 6:
    print('Die back')
    print('This is not a disease,This is just a dead leaf.')
elif predicted_class == 7:
    print('Downy Mildew')
    print('''
Description:
          On the foliage, downy mildew symptoms begin as small, water-soaked spots. Lesions first appear slightly chlorotic, with a yellow-green appearance and progress to a bright yellow on the upper leaf surface. As lesions progress, they become angular, brown (necrotic) and distorted, and plants may defoliate
Condition of Infection:
          Fungus-like organisms.Downy mildew is a disease caused by Oomycetes, which attack plants like cucurbits, causing yellowing, molding, and leaf death. It is host-specific and can cause severe crop reduction. Commonly affected plants include basil, watermelon, zucchini, winter squashes, cucumbers, and pumpkins. The disease spreads favored by warm and humid conditions, allowing for elevated spore production. It can be tracked using Cornell University's Integrated Pest Management Pest Information Platform for Extension and Education. Symptoms can appear in late spring or August.
Mechanical treatment:
          Plant resistant cultivars, remove infected foliage, avoid crowding plants, and water in the morning. No fungicides are available.
Chemical treatment:
          Protective fungicides like Chlorothalonil and copper-based compounds protect crops from early infections, while eradicative fungicides like broad-spectrum systemic fungicides and neem oil are used for severe infections.
''')
elif predicted_class == 8:
    print('Healthy')
    print('This is not a disease,This is just a Healthy plant leaf.')
elif predicted_class == 9:
    print('Leaf Blight')
    print('''
Description: 
          Leaf blight is a general term for diseases that cause browning and death of leaf tissue. It can be caused by fungi, bacteria, or environmental factors.
Conditions of Infection:
          Wet and humid conditions favor the development and spread of leaf blight pathogens.
          Fungal or bacterial pathogens are often the primary cause of leaf blight.
          Poor air circulation and overcrowding can exacerbate the problem.
Mechanical treatment:
          Remove and destroy affected leaves to reduce sources of inoculum.
          Improve air circulation around plants by proper spacing and pruning.
          Avoid overhead watering and water early in the day to allow foliage to dry before nightfall.
Chemical Treatment:
          Fungicides containing chlorothalonil, mancozeb, or copper-based compounds can be effective against fungal blight.
          Bactericides may be needed if bacterial blight is present.
Specific Chemical Treatments:
          Daconil (chlorothalonil): Provides broad-spectrum control against many fungal pathogens.
          Mancozeb: Effective against a wide range of fungal diseases.
          Copper fungicides (e.g., Bordeaux mixture): Effective against many fungal and bacterial diseases.
          Streptomycin: An antibiotic effective against bacterial blight.
''')
elif predicted_class == 10:
    print('Leaf Necrosis')
    print('''
Description: 
          Leaf necrosis refers to the death of leaf tissue, often presenting as brown or black spots or patches. It can result from fungal or bacterial infections, chemical damage, or environmental stress.
Conditions of Infection:
          Fungal or bacterial infections can cause necrotic spots or patches on leaves.
          Chemical damage from herbicides, pesticides, or fertilizers can lead to necrosis.
          Environmental stress, such as drought, extreme temperatures, or nutrient imbalances, can cause leaf tissue to die.
Mechanical Treatment:
          Remove and destroy affected plant parts to reduce sources of infection.
          Improve overall plant health through proper care, including adequate watering, nutrition, and protection from extreme conditions.
Chemical Treatment:
          Fungicides or bactericides may be needed, depending on the cause of necrosis.
Specific Chemical Treatments:
          Copper-based fungicides (e.g., Bordeaux mixture): Effective against many fungal and bacterial diseases.
          Streptomycin: An antibiotic effective against bacterial infections causing necrosis.
''')
elif predicted_class == 11:
    print('Leaf Spot')
    print('''
Description:
          Leaf spot is a general term used to describe a variety of diseases caused by different pathogens such as fungi, bacteria, and sometimes viruses. The disease is characterized by spots on the leaves that can vary in size, shape, and color depending on the causative agent. The spots are typically round and can be brown, black, or yellow with a darker margin. Severe infections can lead to premature leaf drop, reducing the plant's vigor and yield.
Conditions of Infection:
          Humidity and Moisture: High humidity and prolonged leaf wetness create an ideal environment for the development and spread of leaf spot pathogens.
          Temperature: Moderate temperatures (15-30°C) are typically conducive to leaf spot infections, though this can vary depending on the specific pathogen.
          Crowded Plantings: Dense foliage and poor air circulation can trap moisture and increase the risk of infection.
          Infected Plant Debris: Pathogens can survive in infected plant debris and soil, facilitating the spread of the disease.
Mechanical Treatment:
          Sanitation: Remove and destroy infected leaves and plant debris to reduce sources of inoculum.
          Watering: Water plants early in the day to allow leaves to dry quickly, and avoid overhead irrigation.
          Spacing: Ensure adequate spacing between plants to improve air circulation.
          Resistant Varieties: Plant disease-resistant varieties when available.
Chemical Treatment:
          Copper-Based Fungicides: Effective against many types of leaf spot diseases. Apply according to the manufacturer's instructions, usually before symptoms appear or at the first sign of disease.
          Chlorothalonil: A broad-spectrum fungicide that can control leaf spot diseases. Follow label directions for application rates and timing.
          Mancozeb: Another fungicide that is effective against leaf spot diseases. Apply as a preventative treatment or when symptoms first appear.  
Specific Chemical Treatments:
          Bacterial Leaf Spot: For bacterial infections, copper-based fungicides can be effective. Additionally, some antibiotics like streptomycin are sometimes used, though they are less common.
          Fungal Leaf Spot: Common fungicides like mancozeb, chlorothalonil, and thiophanate-methyl can be effective. Regular applications during the growing season may be necessary, particularly in humid or wet conditions.
''')
elif predicted_class == 12:
    print('Mosaic Virus')
    print('''
Description:
          any virus that causes infected plant foliage to have a mottled appearance. Such viruses come from a variety of unrelated lineages and consequently there is no taxon that unites all mosaic viruses.
Cause of Infection: 
          Tomato mosaic virus and tobacco mosaic virus.   Mosaic viruses are plant viruses that cause mottling, discoloration, and distortion of leaves in infected plants. They can be transmitted by aphids or contaminated seeds or cuttings, affecting various crops like roses, beans, tomatoes, potatoes, cucumbers, pumpkins, squash, melons, and peppers. Symptoms include yellow, white, or green stripes, wrinkled leaves, yellowing veins, stunted growth, mottled fruit, and dark green blisters on stems.
Mechanical Treatment: 
          No cure, but prevention involves removing infected plants and avoiding spread.
Chemical Treatment:  
          Mosaic viruses, which affect crops like tomatoes, cucumbers, and peppers, are uncurable due to lack of chemical treatments. Preventive measures include maintaining garden hygiene, promptly removing infected plants, and controlling insect vectors like aphids to reduce transmission.
''')
elif predicted_class == 13:
    print('Powdery Mildew')
    print('''
Description:
          mildew, a conspicuous mass of white threadlike hyphae and fruiting structures produced by various fungi. Mildew is commonly associated with damp cloth, fibres, leather goods, and several plant diseases (downy mildew and powdery mildew)
Cause of Infection: 
          A fungus.Powdery mildew is a fungal disease affecting various plant species, including trees, shrubs, vines, flowers, vegetables, fruits, grasses, field crops, and weeds. Key factors include warm and humid conditions, crowded plantings, high relative humidity, low soil moisture, and young plants. Monitoring gardens during warm and humid periods and taking preventive measures can help keep powdery mildew at bay.
Mechanical Treatment:
          Remove infected leaves, ensure good drainage, and avoid overhead watering at night. Commercial fungicides or a baking soda solution can help.
Chemical Treatment: 
          Apply fungicides like potassium bicarbonate, neem oil, sulfur, or copper to control powdery mildew. Remove infected parts with clippers. Improve garden management for early intervention.
''')

elif predicted_class == 14:
    print('Rust')
    print('''
Description:
          Rust is a common fungal disease affecting a wide range of plants, including roses, beans, and grasses. The disease is characterized by small, yellow to orange pustules on the undersides of leaves, which can eventually turn black. These pustules release spores that spread the infection. In severe cases, rust can cause leaves to wither and die, reducing the plant's vigor and yield.
Conditions of Infection:
          Humidity and Moisture: High humidity and prolonged periods of leaf wetness are ideal conditions for rust fungi to thrive and spread.
          Temperature: Moderate temperatures (15-25°C) are generally conducive to rust infections.
          Air Circulation: Poor air circulation can trap moisture around the plant, promoting the development of rust.
          Susceptible Plant Varieties: Some plant varieties are more susceptible to rust infections than others.
Mechanical Treatment:
          Sanitation: Remove and destroy infected leaves and plant debris to reduce sources of inoculum.
          Watering: Water plants early in the day to allow leaves to dry quickly, and avoid overhead irrigation.
          Spacing: Ensure adequate spacing between plants to improve air circulation.
          Resistant Varieties: Plant disease-resistant varieties when available.
Chemical Treatments:
          Sulfur-Based Fungicides: Effective against rust diseases. Apply according to the manufacturer's instructions, usually before symptoms appear or at the first sign of disease.
          Copper-Based Fungicides: Can help control rust diseases. Follow label directions for application rates and timing.
          Myclobutanil: A systemic fungicide that can protect new growth from rust infections. Apply as a preventative treatment or when symptoms first appear.   
Specific Chemical Treatments:
          Fungicide Sprays: Use fungicides containing sulfur, copper, or myclobutanil to control rust. Regular applications during the growing season may be necessary, particularly in humid or wet conditions.
          Rotation of Fungicides: To prevent the development of resistance, rotate between different fungicides with different modes of action.     
''')

elif predicted_class == 15:
    print('Septoria Leaf Spot')
    print('''
Description:
          Septoria leaf spot is a fungal disease caused by the pathogen Septoria lycopersici, commonly affecting tomato plants. It is characterized by small, water-soaked spots that turn brown and are surrounded by a yellow halo. The spots eventually coalesce, causing significant damage to the foliage. Severe infections can lead to defoliation, reducing the plant’s ability to photosynthesize and thus diminishing fruit yield and quality.
Conditions of Infection:
          Humidity and Moisture: High humidity and prolonged periods of leaf wetness favor the development and spread of Septoria leaf spot.
          Temperature: Moderate temperatures (20-25°C) are optimal for the growth and infection of Septoria lycopersici.
          Plant Debris: The pathogen overwinters in infected plant debris and soil, facilitating infection during the growing season.
          Overhead Irrigation: Splashing water can spread the spores from infected to healthy leaves.
Mechanical Treatment:
          Sanitation: Remove and destroy infected leaves and plant debris to reduce sources of inoculum.
          Watering: Water plants at the base to keep foliage dry and avoid overhead irrigation.
          Crop Rotation: Rotate crops with non-host plants to reduce soilborne inoculum.
          Resistant Varieties: Plant disease-resistant tomato varieties when available.
Chemical Treatments:
          Chlorothalonil: A broad-spectrum fungicide effective against Septoria leaf spot. Apply according to the manufacturer's instructions, typically every 7-10 days during favorable conditions for the disease.
          Copper-Based Fungicides: Effective against a range of fungal diseases, including Septoria leaf spot. Follow label directions for application rates and timing.
          Mancozeb: Another fungicide effective in controlling Septoria leaf spot. Regular applications during the growing season may be necessary.
Specific Chemical Treatments:
          Fungicide Sprays: Use fungicides such as chlorothalonil, copper-based fungicides, or mancozeb as a preventative measure or at the first sign of disease. Reapply as per the manufacturer's instructions, usually every 7-10 days, especially in humid conditions.
          Rotation of Fungicides: To prevent the development of resistance, rotate between different fungicides with different modes of action.
''')

elif predicted_class == 16:
    print('Sooty mold')
    print('''
Description:
          Sooty mold is a fungal disease that grows on plants and other surfaces covered by honeydew, a sticky substance created by certain insects. Sooty mold's name comes from the dark threadlike growth (mycelium) of the fungi resembling a layer of soot.
Cause of Infection:
          Sooty mold is a fungal disease that forms a black coating on plants' leaves and stems, blocking sunlight and reducing photosynthesis. It can also diminish a plant's aesthetic value. 
Mechanical Treatment: 
          To combat sooty mold, identify the source, eliminate the pest, wash off the mold, and consider using neem oil as a treatment. Although sooty mold isn't lethal, immediate action is crucial to prevent further damage.
Chemical Treatment: 
          Chemical treatment for sooty mold involves washing off mold with a hosepipe, controlling insect vectors with contact insecticides, and preventing infestations by regularly cleaning foliage and managing pests.
''')

elif predicted_class == 17:
    print('Sooty Mould')
    print('''
Description:
          Sooty mould is a fungal disease characterized by the presence of black, soot-like deposits on the surface of leaves, stems, and fruits of plants. It is not a parasitic fungus; instead, it grows on the honeydew excreted by sap-sucking insects such as aphids, whiteflies, scale insects, and mealybugs. The black coating can interfere with photosynthesis by blocking sunlight, leading to reduced plant vigor and growth.
Cause of Infection:
          Sooty mould develops on the honeydew secretions from sap-sucking insects. These insects feed on plant sap and excrete a sugary substance called honeydew, which creates a favorable environment for the growth of sooty mould fungi. The primary factors contributing to sooty mould infection include:
          Presence of Sap-Sucking Insects: Aphids, whiteflies, scale insects, and mealybugs are common culprits.
          High Humidity: Moist conditions favor the growth of sooty mould fungi.
          Poor Air Circulation: Dense foliage and overcrowded plantings can contribute to an increase in humidity and moisture retention.
Mechanical Treatment:
          Pruning: Regularly prune and thin out plants to improve air circulation and reduce humidity.
          Washing: Wash affected plants with a strong stream of water to remove both the honeydew and the sooty mould.
          Insect Control: Manage and control the population of sap-sucking insects using natural predators or insecticidal soaps to prevent honeydew formation.
          Cleaning: Remove and destroy heavily infested plant parts to reduce sources of honeydew and sooty mould.
Chemical Treatment:
          Insecticidal Soaps: Effective in controlling sap-sucking insects that produce honeydew. Apply according to label instructions, ensuring thorough coverage of the plant.
          Horticultural Oils: These oils can suffocate insects and reduce honeydew production. Apply during cooler parts of the day to avoid leaf burn.
          Systemic Insecticides: These are absorbed by the plant and can provide longer-lasting protection against sap-sucking insects. Follow label directions for proper use.
Specific Chemical Treatments:
Neem Oil:
          A natural oil that can control a wide range of sap-sucking insects.
          Apply every 7-14 days, depending on the severity of the infestation.
Imidacloprid:
          A systemic insecticide that is effective against sap-sucking insects.
          Apply as a soil drench or foliar spray according to label instructions.
Pyrethrin-Based Insecticides:
          Effective for quick knockdown of insects like aphids and whiteflies.
          Use as a contact spray and reapply as needed based on label instructions.
Horticultural Oil (Petroleum-based or Plant-based):
          Controls insects by suffocation and can reduce sooty mould by eliminating honeydew production.
          Apply during dormant season for best results, and follow label directions carefully to avoid plant damage.
''')