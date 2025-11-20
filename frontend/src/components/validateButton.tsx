import cx from 'classnames';
import { FC } from 'react';
import { GrValidate } from 'react-icons/gr';

import { useComputeModelPrediction } from '../core/api';
import { CSSProperties } from 'styled-components';

interface validateButtonsProps {
  projectSlug: string | null;
  modelName: string | null;
  kind: string | null;
  currentScheme: string | null;
  className?: string;
  id?: string;
  buttonLabel?: string;
  style?: CSSProperties;
}

export const ValidateButtons: FC<validateButtonsProps> = ({
  modelName,
  kind,
  currentScheme,
  projectSlug,
  className,
  id,
  buttonLabel,
  style,
}) => {
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, 16);
  return (
    <button
      className={cx(className ? className : 'btn btn-primary')}
      style={style ? style : { color: 'white' }}
      onClick={() => {
        computeModelPrediction(modelName || '', 'annotable', currentScheme, kind);
      }}
      id={id}
    >
      <GrValidate size={20} /> {buttonLabel ? buttonLabel : 'Compute statistics on annotations'}
    </button>
  );
};
