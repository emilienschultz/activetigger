import { FC, useEffect, useState } from 'react';
import { MLStatisticsModel } from '../types';
import { DisplayScores } from './DisplayScores';

interface Scores {
  [key: string]: MLStatisticsModel;
}

interface DisplayScoresMenuPropos {
  projectSlug?: string | null;
  scores: Scores;
  modelName?: string;
  displayTitle?: boolean;
  skip?: string[];
}

export const DisplayScoresMenu: FC<DisplayScoresMenuPropos> = ({
  scores,
  modelName,
  displayTitle,
  skip,
  projectSlug,
}) => {
  const allowedScores = Object.entries(scores)
    .filter(([_, value]) => value != null)
    .filter(([key]) => !skip?.includes(key));
  const scoreKeys = allowedScores.map(([key]) => key);
  const [currentScore, setCurrentScore] = useState<string>(scoreKeys[0] || '');
  // Ensure currentScore is still valid when scores change
  useEffect(() => {
    if (!scoreKeys.includes(currentScore)) {
      setCurrentScore(Object.keys(scoreKeys)[0] || '');
    }
  }, [scores, currentScore, allowedScores, scoreKeys]);

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
          {allowedScores.map(([key]) => (
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
          projectSlug={projectSlug}
        />
      )}
    </div>
  );
};
