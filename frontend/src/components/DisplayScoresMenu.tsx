import { FC, useEffect, useState } from 'react';
import { MLStatisticsModel } from '../types';
import { DisplayScores } from './DisplayScores';

interface Scores {
  [key: string]: MLStatisticsModel;
}

interface DisplayScoresMenuPropos {
  scores: Scores;
  modelName?: string;
  displayTitle?: boolean;
}

export const DisplayScoresMenu: FC<DisplayScoresMenuPropos> = ({
  scores,
  modelName,
  displayTitle,
}) => {
  const keys = Object.keys(scores);
  const [currentScore, setCurrentScore] = useState<string>(keys[0]);

  // Ensure currentScore is still valid when scores change
  useEffect(() => {
    if (!keys.includes(currentScore)) {
      setCurrentScore(keys[0] || '');
    }
  }, [scores, currentScore, keys]);

  if (!scores || Object.keys(scores).length === 0) {
    return <div>No scores available</div>;
  }

  console.log(currentScore);

  return (
    <div>
      <label htmlFor="statistics">
        Scores{' '}
        <select
          id="statistics"
          value={currentScore}
          onChange={(e) => setCurrentScore(e.target.value)}
        >
          {Object.entries(scores)
            .filter(([_, value]) => value != null)
            .map(([key]) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
        </select>
      </label>
      {scores && (
        <DisplayScores
          title={displayTitle ? currentScore : null}
          scores={scores[currentScore]}
          modelName={modelName}
        />
      )}
    </div>
  );
};
