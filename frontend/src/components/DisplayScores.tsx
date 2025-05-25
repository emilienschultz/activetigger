import { FC } from 'react';
import { MLStatisticsModel } from '../types';
import { DisplayTable } from './DisplayTableStatistics';

export interface DisplayScores {
  title: string;
  scores: MLStatisticsModel;
  scores_cv10?: MLStatisticsModel;
}

// component
export const DisplayScores: FC<DisplayScores> = ({ title, scores, scores_cv10 }) => {
  return (
    <div>
      <DisplayTable scores={scores} title={title} />
      {scores_cv10 && <DisplayTable scores={scores_cv10} title="Cross validation" />}
    </div>
  );
};
