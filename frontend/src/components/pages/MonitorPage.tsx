import { FC } from 'react';
import { useGetQueue } from '../../core/api';
import { PageLayout } from '../layout/PageLayout';

export const MonitorPage: FC = () => {
  const { queueState, gpu } = useGetQueue(null);

  return (
    <PageLayout currentPage="monitor">
      <div className="container-fluid">
        <div className="row">{JSON.stringify(queueState)}</div>
      </div>{' '}
    </PageLayout>
  );
};
