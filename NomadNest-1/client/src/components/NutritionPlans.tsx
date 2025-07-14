import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { NutritionPlan } from "@shared/schema";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2 } from "lucide-react";

const NutritionPlans = () => {
  const { data: nutritionPlans, isLoading, error } = useQuery<NutritionPlan[]>({
    queryKey: ["/api/nutrition-plans"],
  });

  // Get color for calorie type badge
  const getCalorieTypeColor = (calorieType: string) => {
    switch (calorieType) {
      case "Calorie Deficit":
        return "bg-green-500/20 text-green-500";
      case "Calorie Surplus":
        return "bg-blue-500/20 text-blue-500";
      case "Calorie Balance":
        return "bg-amber-500/20 text-amber-500";
      default:
        return "bg-gray-500/20 text-gray-500";
    }
  };

  if (error) {
    return (
      <section id="nutrition" className="mb-12 scroll-mt-16">
        <div className="text-center mb-8">
          <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
            Nutrition Recommendations
          </h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Fuel your body with the right nutrition based on your body type and fitness goals.
          </p>
        </div>
        <p className="text-center text-red-500">Failed to load nutrition plans. Please refresh the page.</p>
      </section>
    );
  }

  return (
    <section id="nutrition" className="mb-12 scroll-mt-16">
      <div className="text-center mb-8">
        <h2 className="font-heading font-bold text-2xl md:text-3xl text-gray-800 mb-2">
          Nutrition Recommendations
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Fuel your body with the right nutrition based on your body type and fitness goals.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {isLoading ? (
          // Skeleton loaders for nutrition plans
          Array(3).fill(0).map((_, i) => (
            <Card key={i} className="overflow-hidden">
              <Skeleton className="h-48 w-full" />
              <CardContent className="p-5">
                <div className="flex justify-between items-center mb-4">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-6 w-24 rounded-full" />
                </div>
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-full mb-4" />
                <Skeleton className="h-10 w-full" />
              </CardContent>
            </Card>
          ))
        ) : (
          nutritionPlans?.map((plan) => (
            <Card key={plan.id} className="overflow-hidden">
              <div className="h-48 bg-gray-200 relative overflow-hidden">
                <img
                  src={plan.imageUrl}
                  alt={plan.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent flex items-end p-4">
                  <h3 className="font-heading font-semibold text-xl text-white">
                    {plan.title}
                  </h3>
                </div>
              </div>
              <CardContent className="p-5">
                <div className="flex justify-between items-center mb-4">
                  <span className="text-sm text-gray-500">
                    Ideal for {plan.bodyTypeId === 1 ? "Ectomorphs" : 
                              plan.bodyTypeId === 2 ? "Mesomorphs" : 
                              plan.bodyTypeId === 3 ? "Endomorphs" : "All Body Types"}
                  </span>
                  <Badge variant="outline" className={getCalorieTypeColor(plan.calorieType)}>
                    {plan.calorieType}
                  </Badge>
                </div>
                <ul className="space-y-2 mb-4">
                  <li className="flex items-start">
                    <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>High protein intake ({plan.proteinPercentage}% of calories)</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Moderate carbs ({plan.carbPercentage}% of calories)</span>
                  </li>
                  <li className="flex items-start">
                    <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Moderate healthy fats ({plan.fatPercentage}% of calories)</span>
                  </li>
                </ul>
                <Button className="w-full" asChild>
                  <Link href={`/meal-plan/${plan.id}`}>
                    Get Meal Plan
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

export default NutritionPlans;
