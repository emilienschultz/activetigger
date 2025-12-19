import classNames from 'classnames';
import { reverse } from 'lodash';
import { FC } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useAppContext } from '../../core/context';

export const AnnotationHistoryList: FC = () => {
  const { appContext } = useAppContext();
  const { projectName, elementId } = useParams();

  const { history } = appContext;

  return (
    <div>
      <h4>Annotation history</h4>
      <ul>
        {reverse(
          history.map((previousElementId, i) => {
            return (
              <li key={`${previousElementId}-${i}`}>
                <Link
                  className={classNames(elementId === previousElementId && 'fw-bold')}
                  to={`/projects/${projectName}/tag/${previousElementId}`}
                >
                  {previousElementId}
                </Link>
              </li>
            );
          }),
        )}
      </ul>
    </div>
  );
};
