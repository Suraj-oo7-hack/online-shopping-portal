import { Card, CardContent } from "@/components/ui/card";

const Education = () => {
  return (
    <section id="education" className="mb-12 scroll-mt-16">
      <div className="text-center mb-8">
        <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
          Fitness Education
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Learn the fundamentals of fitness and nutrition to make informed decisions about your health journey.
        </p>
      </div>

      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-heading font-semibold text-xl mb-4">Understanding BMI</h3>
              <p className="text-gray-600 mb-4">
                Body Mass Index (BMI) is a measurement that helps assess if someone has a healthy body weight for their height. 
                It's calculated by dividing weight in kilograms by height in meters squared.
              </p>
              
              <div className="mb-4 bg-blue-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">BMI Categories:</h4>
                <ul className="space-y-1">
                  <li className="flex items-center">
                    <span className="w-3 h-3 bg-blue-300 rounded-full mr-2"></span>
                    <span>Below 18.5: Underweight</span>
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 bg-green-400 rounded-full mr-2"></span>
                    <span>18.5 - 24.9: Normal weight</span>
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></span>
                    <span>25.0 - 29.9: Overweight</span>
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 bg-orange-400 rounded-full mr-2"></span>
                    <span>30.0 - 34.9: Obesity (Class 1)</span>
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 bg-red-400 rounded-full mr-2"></span>
                    <span>35.0 and above: Obesity (Class 2 & 3)</span>
                  </li>
                </ul>
              </div>
              
              <p className="text-gray-600">
                While BMI is a useful screening tool, it doesn't directly measure body fat or account for factors like muscle mass, 
                bone density, or overall body composition.
              </p>
            </div>
            
            <div>
              <h3 className="font-heading font-semibold text-xl mb-4">Body Types Explained</h3>
              <p className="text-gray-600 mb-4">
                Understanding your body type (somatotype) can help you create more effective fitness and nutrition strategies.
              </p>
              
              <div className="space-y-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">Ectomorph</h4>
                  <p className="text-sm text-gray-600">
                    Naturally thin with smaller bone structure and lean muscle. Typically have fast metabolism and difficulty gaining weight.
                  </p>
                  <p className="text-sm font-medium mt-1">
                    Focus: Muscle building with strength training and higher calorie intake
                  </p>
                </div>
                
                <div className="bg-amber-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">Mesomorph</h4>
                  <p className="text-sm text-gray-600">
                    Athletic build with medium bone structure. Gain muscle easily and have a responsive metabolism.
                  </p>
                  <p className="text-sm font-medium mt-1">
                    Focus: Balanced approach with both strength and cardio
                  </p>
                </div>
                
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-medium mb-1">Endomorph</h4>
                  <p className="text-sm text-gray-600">
                    Softer, rounder physique with larger bone structure. Tend to store fat easily and may have slower metabolism.
                  </p>
                  <p className="text-sm font-medium mt-1">
                    Focus: Higher protein intake with emphasis on cardio and strength training
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </section>
  );
};

export default Education;
