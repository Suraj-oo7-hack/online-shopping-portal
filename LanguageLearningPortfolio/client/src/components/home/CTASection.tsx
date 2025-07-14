import { Link } from "wouter";

export default function CTASection() {
  return (
    <section className="py-16 bg-[#88B2D3]">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="font-quicksand font-bold text-2xl md:text-4xl text-white mb-4">
            Ready to Transform Your Images?
          </h2>
          <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
            Join thousands of users creating magical Ghibli-inspired artwork with our AI-powered platform
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link href="#transform">
              <a className="font-quicksand font-semibold bg-white text-[#88B2D3] hover:bg-[#F7F3E9] transition-colors rounded-full px-8 py-3 shadow-md">
                Start Transforming Now
              </a>
            </Link>
            <Link href="#features">
              <a className="font-quicksand font-semibold border-2 border-white text-white hover:bg-white/10 transition-colors rounded-full px-8 py-3">
                Learn More
              </a>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
