const Hero = () => {
  return (
    <section className="relative h-64 md:h-80 bg-gradient-to-r from-primary to-blue-700 text-white overflow-hidden">
      <div className="absolute inset-0 bg-black opacity-20"></div>
      <div className="container mx-auto px-4 h-full flex items-center relative z-10">
        <div className="max-w-2xl">
          <h1 className="font-heading font-bold text-3xl md:text-4xl lg:text-5xl mb-4">
            Your Journey to a Healthier You
          </h1>
          <p className="text-lg md:text-xl">
            Calculate your BMI, discover personalized workouts and nutrition plans based on your body type.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Hero;
