import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Heart } from "lucide-react";
import { Button } from "@/components/ui/button";

type GalleryFilter = "all" | "landscapes" | "portraits" | "animals" | "architecture";

type GalleryItem = {
  id: string;
  imageUrl: string;
  title: string;
  style: string;
  userName: string;
  userAvatar: string;
  likes: number;
};

export default function GallerySection() {
  const [filter, setFilter] = useState<GalleryFilter>("all");
  
  const { data: galleryItems, isLoading } = useQuery({
    queryKey: ["/api/gallery"],
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
  
  return (
    <section id="gallery" className="py-16 md:py-24 bg-[#F7F3E9]/70 relative">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="font-quicksand font-bold text-2xl md:text-4xl text-[#3C4F65] text-center mb-4">
            Explore Our Gallery
          </h2>
          <p className="text-center text-[#3C4F65]/70 max-w-2xl mx-auto mb-12">
            Get inspired by these amazing transformations from our community
          </p>
          
          {/* Filter Tabs */}
          <div className="flex flex-wrap justify-center gap-3 mb-10">
            <Button
              onClick={() => setFilter("all")}
              className={`px-6 py-2 rounded-full font-quicksand font-medium ${
                filter === "all"
                  ? "bg-[#88B2D3] text-white"
                  : "bg-white text-[#3C4F65]"
              }`}
            >
              All Styles
            </Button>
            <Button
              onClick={() => setFilter("landscapes")}
              className={`px-6 py-2 rounded-full font-quicksand font-medium ${
                filter === "landscapes"
                  ? "bg-[#88B2D3] text-white"
                  : "bg-white text-[#3C4F65]"
              }`}
            >
              Landscapes
            </Button>
            <Button
              onClick={() => setFilter("portraits")}
              className={`px-6 py-2 rounded-full font-quicksand font-medium ${
                filter === "portraits"
                  ? "bg-[#88B2D3] text-white"
                  : "bg-white text-[#3C4F65]"
              }`}
            >
              Portraits
            </Button>
            <Button
              onClick={() => setFilter("animals")}
              className={`px-6 py-2 rounded-full font-quicksand font-medium ${
                filter === "animals"
                  ? "bg-[#88B2D3] text-white"
                  : "bg-white text-[#3C4F65]"
              }`}
            >
              Animals
            </Button>
            <Button
              onClick={() => setFilter("architecture")}
              className={`px-6 py-2 rounded-full font-quicksand font-medium ${
                filter === "architecture"
                  ? "bg-[#88B2D3] text-white"
                  : "bg-white text-[#3C4F65]"
              }`}
            >
              Architecture
            </Button>
          </div>
          
          {/* Gallery Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {isLoading ? (
              // Loading skeleton
              Array.from({ length: 6 }).map((_, index) => (
                <div key={index} className="bg-white rounded-2xl overflow-hidden shadow-md animate-pulse">
                  <div className="aspect-[4/3] bg-gray-200"></div>
                  <div className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 rounded-full bg-gray-200"></div>
                        <div className="h-4 w-20 bg-gray-200 rounded"></div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="h-4 w-4 bg-gray-200 rounded"></div>
                        <div className="h-4 w-8 bg-gray-200 rounded"></div>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : !galleryItems || galleryItems.length === 0 ? (
              <div className="col-span-3 text-center py-12">
                <p className="text-[#3C4F65]/80 mb-4">No gallery items found.</p>
                <p className="text-sm text-[#3C4F65]/60">
                  Be the first to transform and share your images!
                </p>
              </div>
            ) : (
              // This will be populated by the API data when available
              // For now showing some example items with URLs from the design
              <>
                <GalleryItemCard 
                  id="1"
                  imageUrl="https://images.unsplash.com/photo-1578913071922-bb54e6a97627?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Mountain Spirits"
                  style="Totoro style"
                  userName="Sarah K."
                  userAvatar="https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={142}
                />
                <GalleryItemCard 
                  id="2"
                  imageUrl="https://images.unsplash.com/photo-1513623935135-c896b59073c1?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Forest Guardian"
                  style="Mononoke style"
                  userName="Michael T."
                  userAvatar="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={97}
                />
                <GalleryItemCard 
                  id="3"
                  imageUrl="https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Valley of Dreams"
                  style="Spirited Away style"
                  userName="Jessica L."
                  userAvatar="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={218}
                />
                <GalleryItemCard 
                  id="4"
                  imageUrl="https://images.unsplash.com/photo-1490730141103-6cac27aaab94?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Sunset Harbor"
                  style="Kiki's style"
                  userName="David W."
                  userAvatar="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={184}
                />
                <GalleryItemCard 
                  id="5"
                  imageUrl="https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Forest Guardian"
                  style="Totoro style"
                  userName="Emma R."
                  userAvatar="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={126}
                />
                <GalleryItemCard 
                  id="6"
                  imageUrl="https://images.unsplash.com/photo-1515578706925-0dc1a7bfc8cb?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  title="Enchanted City"
                  style="Howl's style"
                  userName="Thomas H."
                  userAvatar="https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=80&auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
                  likes={153}
                />
              </>
            )}
          </div>
          
          {/* Load More Button */}
          <div className="text-center mt-12">
            <Button
              variant="outline"
              className="font-quicksand font-medium text-[#88B2D3] border-2 border-[#88B2D3] hover:bg-[#88B2D3]/10 rounded-full px-8 py-3"
            >
              Load More Images
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}

function GalleryItemCard({ 
  id, 
  imageUrl, 
  title, 
  style, 
  userName, 
  userAvatar, 
  likes 
}: GalleryItem) {
  const [isLiked, setIsLiked] = useState(false);
  const [likeCount, setLikeCount] = useState(likes);
  
  const toggleLike = () => {
    if (isLiked) {
      setLikeCount(prev => prev - 1);
    } else {
      setLikeCount(prev => prev + 1);
    }
    setIsLiked(!isLiked);
  };
  
  return (
    <div className="group bg-white rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition-all transform hover:-translate-y-1">
      <div className="relative aspect-[4/3] overflow-hidden">
        <img 
          src={imageUrl} 
          alt={title} 
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
          <div>
            <h3 className="font-quicksand font-semibold text-white">{title}</h3>
            <p className="text-white/80 text-sm">{style}</p>
          </div>
        </div>
      </div>
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <img 
              src={userAvatar} 
              alt={`${userName} avatar`} 
              className="w-8 h-8 rounded-full object-cover" 
            />
            <span className="text-sm text-[#3C4F65]/80">{userName}</span>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              className={`transition-colors ${
                isLiked ? "text-[#F4A6A1]" : "text-[#3C4F65]/60 hover:text-[#88B2D3]"
              }`}
              onClick={toggleLike}
            >
              <Heart className={`h-4 w-4 ${isLiked ? "fill-[#F4A6A1]" : ""}`} />
            </button>
            <span className="text-sm text-[#3C4F65]/60">{likeCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
