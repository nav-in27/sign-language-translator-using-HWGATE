/**
 * AC-HWGATE Recognizer (ONNX Runtime Web Client)
 * 
 * Takes a sliding window of frames (32 frames of 133 keypoints)
 * and passes them through our AC-HWGATE model to predict ISL Gloss probabilities.
 */

const mockVocab = ["I", "TOMORROW", "MARKET", "GO", "YOU", "HELLO", "HOW", "ARE"];

export const recognizeSign = async (keypointsSequence, confidencesSequence) => {
  // Simulate ONNX execution for the spatio-temporal graph transformer
  await new Promise(resolve => setTimeout(resolve, 35));
  
  // Randomly return a mock gloss for demonstration purposes
  // To avoid rapid flashing in the UI, we only return a gloss occasionally
  if (Math.random() > 0.95) {
    const gloss1 = mockVocab[Math.floor(Math.random() * mockVocab.length)];
    const gloss2 = mockVocab[Math.floor(Math.random() * mockVocab.length)];
    return `${gloss1} ${gloss2}`;
  }
  
  return null;
};
