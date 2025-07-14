import { 
  WandSparkles, 
  Sliders, 
  Download, 
  Images, 
  Smartphone, 
  Share2 
} from "lucide-react";

const features = [
  {
    icon: <WandSparkles className="text-2xl text-[#88B2D3]" />,
    title: "AI-Powered Transformations",
    description: "Our advanced models transform your photos into Studio Ghibli inspired artwork with incredible detail and authenticity."
  },
  {
    icon: <Sliders className="text-2xl text-[#88B2D3]" />,
    title: "Customizable Styles",
    description: "Choose from various Ghibli film styles and adjust settings to create the perfect transformation for your images."
  },
  {
    icon: <Download className="text-2xl text-[#88B2D3]" />,
    title: "High Quality Exports",
    description: "Download your transformed images in high resolution for printing, sharing or using in your projects."
  },
  {
    icon: <Images className="text-2xl text-[#88B2D3]" />,
    title: "Before & After Comparison",
    description: "Compare your original photo with the transformed version using our interactive side-by-side slider view."
  },
  {
    icon: <Smartphone className="text-2xl text-[#88B2D3]" />,
    title: "Fully Responsive",
    description: "Transform your photos anytime, anywhere - our app works perfectly on mobile, tablet, and desktop devices."
  },
  {
    icon: <Share2 className="text-2xl text-[#88B2D3]" />,
    title: "Easy Sharing",
    description: "Share your magical transformations directly to social media or save them to your device with one click."
  }
];

export default function FeaturesSection() {
  return (
    <section id="features" className="py-16 md:py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="font-quicksand font-bold text-2xl md:text-4xl text-[#3C4F65] text-center mb-4">
            Transform With AI Magic
          </h2>
          <p className="text-center text-[#3C4F65]/70 max-w-2xl mx-auto mb-16">
            Our advanced AI models create stunning Ghibli-style transformations with just a few clicks
          </p>
          
          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="bg-[#F7F3E9]/30 rounded-2xl p-6 text-center hover:shadow-md transition-shadow"
              >
                <div className="w-16 h-16 mx-auto mb-5 bg-[#88B2D3]/10 rounded-full flex items-center justify-center">
                  {feature.icon}
                </div>
                <h3 className="font-quicksand font-semibold text-xl text-[#3C4F65] mb-3">
                  {feature.title}
                </h3>
                <p className="text-[#3C4F65]/70">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
