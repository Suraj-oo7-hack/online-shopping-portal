import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { BodyType } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { CheckCircle2 } from "lucide-react";

const BodyTypes = () => {
  const { data: bodyTypes, isLoading, error } = useQuery<BodyType[]>({
    queryKey: ["/api/body-types"],
  });

  if (error) {
    return (
      <section id="body-types" className="mb-12 scroll-mt-16">
        <div className="text-center mb-8">
          <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
            Identify Your Body Type
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Understanding your body type helps in creating optimal workout and nutrition plans.
          </p>
        </div>
        <p className="text-center text-red-500">Failed to load body types. Please refresh the page.</p>
      </section>
    );
  }

  return (
    <section id="body-types" className="mb-12 scroll-mt-16">
      <div className="text-center mb-8">
        <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
          Identify Your Body Type
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Understanding your body type helps in creating optimal workout and nutrition plans.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {isLoading ? (
          // Skeleton loaders for body types
          Array(3).fill(0).map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="h-48 w-full" />
              <CardContent className="p-5">
                <Skeleton className="h-6 w-32 mb-4" />
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-full mb-4" />
                <Skeleton className="h-10 w-full" />
              </CardContent>
            </Card>
          ))
        ) : (
          bodyTypes?.map((bodyType) => (
            <Card key={bodyType.id} className="overflow-hidden">
              <div className="h-48 bg-gray-200 relative overflow-hidden">
                <img 
                  src={bodyType.imageUrl} 
                  alt={`${bodyType.name} body type`} 
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex items-end p-4">
                  <h3 className="font-heading font-semibold text-xl text-white">
                    {bodyType.name}
                  </h3>
                </div>
              </div>
              <CardContent className="p-5">
                <ul className="space-y-2 mb-4">
                  {(bodyType.characteristics as string[]).map((characteristic, index) => (
                    <li key={index} className="flex items-start">
                      <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                      <span>{characteristic}</span>
                    </li>
                  ))}
                </ul>
                <Button 
                  className="w-full"
                  asChild
                >
                  <Link href={`/#workouts?bodyType=${bodyType.id}`}>
                    Get {bodyType.name} Plan
                  </Link>
                </Button>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </section>
  );
};

export default BodyTypes;
