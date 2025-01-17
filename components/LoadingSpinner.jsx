import React from 'react';
const LoadingSpinner = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-[300px]">
      <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin"></div>
      <p className="mt-4 text-lg text-gray-600">Training Neural Network Model...</p>
      <p className="mt-2 text-sm text-gray-500">This may take a few moments</p>
    </div>
  );
};
export default LoadingSpinner;