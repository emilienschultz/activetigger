import { FC, useEffect, useState } from 'react';
import { DisplayScores } from './DisplayScores';

interface DisplayScoresMenuPropos {
  scores: Record<string, Record<string, number>>;
  modelName?: string;
  displayTitle?: boolean;
}

export const DisplayScoresMenu: FC<DisplayScoresMenuPropos> = ({
  scores,
  modelName,
  displayTitle,
}) => {
  const keys = Object.keys(scores || {});
  const [currentScore, setCurrentScore] = useState<string>(keys[0] || '');

  // Ensure currentScore is still valid when scores change
  useEffect(() => {
    if (!keys.includes(currentScore)) {
      setCurrentScore(keys[0] || '');
    }
  }, [scores, currentScore, keys]);

  if (!scores || Object.keys(scores).length === 0) {
    return <div>No scores available</div>;
  }
  return (
    <div>
      <label htmlFor="statistics">
        Scores{' '}
        <select
          id="statistics"
          value={currentScore}
          onChange={(e) => setCurrentScore(e.target.value)}
        >
          {Object.entries(scores).map(([key]) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      </label>
      {scores[currentScore] && (
        <DisplayScores
          title={displayTitle ? currentScore : null}
          scores={scores[currentScore]}
          modelName={modelName}
        />
      )}
    </div>
  );
};
