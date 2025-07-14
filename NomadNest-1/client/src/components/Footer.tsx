import { Dumbbell } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-gray-800 text-white mt-auto">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Dumbbell className="h-6 w-6" />
              <h2 className="font-heading font-bold text-xl">FitTrack</h2>
            </div>
            <p className="text-gray-400">
              Your personalized fitness companion that helps you understand your body and achieve your health goals.
            </p>
          </div>
          
          <div>
            <h3 className="font-heading font-semibold text-lg mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="#bmi-calculator" className="text-gray-400 hover:text-white transition">
                  BMI Calculator
                </a>
              </li>
              <li>
                <a href="#body-types" className="text-gray-400 hover:text-white transition">
                  Body Types
                </a>
              </li>
              <li>
                <a href="#workouts" className="text-gray-400 hover:text-white transition">
                  Workout Plans
                </a>
              </li>
              <li>
                <a href="#nutrition" className="text-gray-400 hover:text-white transition">
                  Nutrition Guide
                </a>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-heading font-semibold text-lg mb-4">Disclaimer</h3>
            <p className="text-gray-400 text-sm">
              The information provided by FitTrack is for general informational purposes only. 
              All information is provided in good faith, however we make no representation or 
              warranty regarding the accuracy, validity or reliability of the information.
            </p>
          </div>
        </div>
        
        <div className="border-t border-gray-700 mt-8 pt-6 text-center text-gray-500 text-sm">
          <p>&copy; {currentYear} FitTrack. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
