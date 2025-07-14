import { Link } from "wouter";

export default function HeroSection() {
  const previewImages = [
    {
      id: 1,
      src: "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
      alt: "Landscape example",
      delay: "0s"
    },
    {
      id: 2,
      src: "https://images.unsplash.com/photo-1500462918059-b1a0cb512f1d?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
      alt: "Ghibli style landscape",
      delay: "2s"
    },
    {
      id: 3,
      src: "https://images.unsplash.com/photo-1578913071922-bb54e6a97627?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
      alt: "Transformed example",
      delay: "4s"
    }
  ];

  return (
    <section className="pt-20 relative overflow-hidden">
      <div className="absolute inset-0 cloud-bg"></div>
      <div className="container mx-auto px-4 py-12 md:py-20 relative z-0">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="font-quicksand font-bold text-3xl md:text-5xl text-[#3C4F65] mb-4">
            Transform Your Images with <span className="text-[#88B2D3]">Ghibli Magic</span>
          </h1>
          <p className="text-lg md:text-xl text-[#3C4F65]/80 mb-8">
            Upload any photo and watch it transform into a beautiful Studio Ghibli inspired artwork using AI magic
          </p>
          <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
            <Link href="#transform">
              <a className="font-quicksand font-semibold bg-[#88B2D3] hover:bg-[#6892B3] text-white rounded-full px-8 py-3 transition-all transform hover:scale-105 shadow-md">
                Start Transforming
              </a>
            </Link>
            <Link href="#gallery">
              <a className="font-quicksand font-semibold border-2 border-[#88B2D3] text-[#88B2D3] hover:bg-[#88B2D3]/10 rounded-full px-8 py-3 transition-all">
                Explore Gallery
              </a>
            </Link>
          </div>
        </div>
      </div>
      <div className="relative max-w-6xl mx-auto px-4 z-0">
        <div className="grid grid-cols-3 gap-3 md:gap-6">
          {previewImages.map((image) => (
            <div
              key={image.id}
              className="transform transition-all hover:-translate-y-2 hover:shadow-xl rounded-2xl overflow-hidden h-40 md:h-64 animate-float"
              style={{ animationDelay: image.delay }}
            >
              <img
                src={image.src}
                alt={image.alt}
                className="w-full h-full object-cover"
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
