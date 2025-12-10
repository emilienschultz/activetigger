import cx from 'classnames';
import { FC } from 'react';
import { GrValidate } from 'react-icons/gr';

import { CSSProperties } from 'styled-components';
import { useComputeModelPrediction } from '../core/api';

interface validateButtonsProps {
  projectSlug: string | null;
  modelName: string | null;
  kind: string | null;
  currentScheme: string | null;
  className?: string;
  id?: string;
  buttonLabel?: string;
  style?: CSSProperties;
  idComputing?: boolean;
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
  idComputing,
}) => {
  const { computeModelPrediction } = useComputeModelPrediction(projectSlug || null, 16);
  return (
    <button
      className={cx(className ? className : 'btn-primary-action')}
      style={style ? style : { color: 'white' }}
      onClick={() => {
        computeModelPrediction(modelName || '', 'annotable', currentScheme, kind);
      }}
      id={id}
      disabled={idComputing}
    >
      <GrValidate size={20} /> {buttonLabel ? buttonLabel : 'Compute statistics on annotations'}
    </button>
  );
};
